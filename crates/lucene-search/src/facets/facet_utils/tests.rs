//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::DocIdSet;

fn it(docs: &[i32]) -> Box<dyn DocIdSetIterator> {
    Box::new(DocIdSet::Array(docs.to_vec()).into_iterator())
}

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
fn live_docs_skips_the_deleted() {
    let mut live = FixedBitSet::new(10);
    for d in [1, 2, 5, 9] {
        // FBS: constant indexes below the bitset's length of 10.
        live.set(d);
    }
    let mut l = live_docs_disi(it(&[0, 1, 3, 5, 6, 9]), &live);
    assert_eq!(docs_of(&mut l), vec![1, 5, 9]);
    let mut l = live_docs_disi(it(&[0, 1, 3, 5, 6, 9]), &live);
    assert_eq!(l.advance(2).unwrap(), 5);
    assert_eq!(l.doc_id(), 5);
    assert_eq!(l.cost(), 6);
    assert_eq!(l.advance(7).unwrap(), 9);
    assert_eq!(l.next_doc().unwrap(), NO_MORE_DOCS);
}

#[test]
fn a_conjunction_has_what_every_iterator_has() {
    let mut c = ConjunctionIterator::new(vec![
        it(&[1, 2, 3, 5, 8, 13, 21]),
        it(&[2, 3, 5, 7, 13, 21, 34]),
        it(&[0, 3, 13, 21]),
    ]);
    assert_eq!(c.doc_id(), -1);
    assert_eq!(c.cost(), 4, "led by the cheapest");
    assert_eq!(docs_of(&mut c), vec![3, 13, 21]);
    assert_eq!(c.doc_id(), NO_MORE_DOCS);

    let mut c = ConjunctionIterator::new(vec![it(&[1, 4, 6, 9]), it(&[4, 5, 6, 9])]);
    assert_eq!(c.advance(5).unwrap(), 6);
    assert_eq!(c.next_doc().unwrap(), 9);

    let mut none = ConjunctionIterator::new(Vec::new());
    assert_eq!(none.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(none.advance(3).unwrap(), NO_MORE_DOCS);
    assert_eq!(none.cost(), 0);
}

#[test]
fn for_each_in_both_visits_the_intersection_in_order() {
    let mut a = DocIdSet::Array(vec![0, 2, 4, 6, 8, 10]).into_iterator();
    let mut b = DocIdSet::Array(vec![1, 2, 3, 8, 9, 10, 11]).into_iterator();
    let mut seen = Vec::new();
    for_each_in_both(&mut a, &mut b, |b| {
        seen.push(b.doc_id());
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, vec![2, 8, 10]);
    // The callback's error stops the walk.
    let mut a = DocIdSet::Array(vec![1, 2]).into_iterator();
    let mut b = DocIdSet::Array(vec![1, 2]).into_iterator();
    let r = for_each_in_both(&mut a, &mut b, |_| {
        Err(crate::Error::IllegalState("stop".into()))
    });
    assert!(r.is_err());
}

/// Doc values on `docs`: positions like a segment's doc-values iterator.
struct Dv {
    docs: Vec<i32>,
    doc: i32,
}

impl DocIdSetIterator for Dv {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        let target = self.doc + 1;
        self.advance(target)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = self
            .docs
            .iter()
            .copied()
            .find(|&d| d >= target)
            .unwrap_or(NO_MORE_DOCS);
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.docs.len() as i64
    }
}

impl DocValuesIterator for Dv {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.doc = target;
        Ok(self.docs.contains(&target))
    }
}

fn hits_of(set: &DocIdSet, dv_docs: &[i32]) -> Vec<i32> {
    let mut dv = Dv {
        docs: dv_docs.to_vec(),
        doc: -1,
    };
    let mut out = Vec::new();
    for_each_hit(set, &mut dv, |dv| {
        out.push(dv.doc_id());
        Ok(())
    })
    .unwrap();
    out
}

fn bits(docs: &[usize], cost: i64) -> DocIdSet {
    let mut bits = FixedBitSet::new(64);
    for &d in docs {
        // FBS: the tests' documents are all below the bitset's length of 64.
        bits.set(d);
    }
    DocIdSet::Bits { bits, cost }
}

#[test]
fn for_each_hit_is_the_conjunction_whichever_side_leads() {
    let dv = [1, 2, 3, 5, 8, 13, 21, 40, 70];
    assert!(hits_of(&DocIdSet::Empty, &dv).is_empty());
    assert_eq!(
        hits_of(&DocIdSet::Array(vec![0, 2, 3, 4, 21]), &dv),
        vec![2, 3, 21]
    );
    let set = [0, 2, 3, 4, 21, 63];
    // A bitset at least as costly as the doc values: they lead, and a
    // document past the bits (70) is not a hit.
    assert_eq!(hits_of(&bits(&set, 9), &dv), vec![2, 3, 21]);
    // A cheaper bitset leads.
    assert_eq!(hits_of(&bits(&set, 6), &dv), vec![2, 3, 21]);
    // An error stops the walk, either way.
    for cost in [6, 9] {
        let mut d = Dv {
            docs: dv.to_vec(),
            doc: -1,
        };
        let r = for_each_hit(&bits(&set, cost), &mut d, |_| {
            Err(crate::Error::IllegalState("stop".into()))
        });
        assert!(r.is_err());
    }
    let mut d = Dv {
        docs: dv.to_vec(),
        doc: -1,
    };
    let r = for_each_hit(&DocIdSet::Array(vec![2]), &mut d, |_| {
        Err(crate::Error::IllegalState("stop".into()))
    });
    assert!(r.is_err());
}

/// A chunk's walk is the whole walk cut to the chunk, whichever side leads,
/// at word edges and inside words, for every kind of hit set.
#[test]
fn for_each_hit_in_is_for_each_hit_cut_to_the_range() {
    let dv: Vec<i32> = (0..250).filter(|d| d % 3 != 1).collect();
    let set: Vec<usize> = (0..250).filter(|d| d % 5 != 2).collect();
    let mut wide = FixedBitSet::new(250);
    for &d in &set {
        // FBS: every document of `set` is below the bitset's length of 250.
        wide.set(d);
    }
    let array = DocIdSet::Array(set.iter().map(|&d| d as i32).collect());
    let led_by_dv = DocIdSet::Bits {
        bits: wide.clone(),
        cost: 1000,
    };
    let led_by_bits = DocIdSet::Bits {
        bits: wide,
        cost: 1,
    };
    let ends = [-5, 0, 1, 63, 64, 65, 100, 127, 128, 200, 249, 250, 400];
    for hits in [&array, &led_by_dv, &led_by_bits, &DocIdSet::Empty] {
        let whole = hits_of(hits, &dv);
        for &lo in &ends {
            for &hi in &ends {
                let mut d = Dv {
                    docs: dv.clone(),
                    doc: -1,
                };
                let mut got = Vec::new();
                for_each_hit_in(hits, &mut d, lo, hi, |d| {
                    got.push(d.doc_id());
                    Ok(())
                })
                .unwrap();
                let want: Vec<i32> = whole
                    .iter()
                    .copied()
                    .filter(|&x| x >= lo && x < hi)
                    .collect();
                assert_eq!(got, want, "{lo}..{hi}");
            }
        }
        // An error stops the walk.
        if !matches!(hits, DocIdSet::Empty) {
            let mut d = Dv {
                docs: dv.clone(),
                doc: -1,
            };
            let r = for_each_hit_in(hits, &mut d, 10, 200, |_| {
                Err(crate::Error::IllegalState("stop".into()))
            });
            assert!(r.is_err());
        }
    }
}

#[test]
fn set_bits_pop_word_by_word_in_order_and_stop_at_an_error() {
    let mut bits = FixedBitSet::new(200);
    for d in [0, 1, 63, 64, 65, 127, 128, 199] {
        // FBS: constant indexes below the bitset's length of 200.
        bits.set(d);
    }
    let mut seen = Vec::new();
    try_for_each_set_bit(&bits, |d| {
        seen.push(d);
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, vec![0, 1, 63, 64, 65, 127, 128, 199]);
    let mut seen = Vec::new();
    let err = try_for_each_set_bit(&bits, |d| {
        seen.push(d);
        if d == 64 {
            return Err(Error::IllegalState("stop".into()));
        }
        Ok(())
    });
    assert!(err.is_err());
    assert_eq!(seen, vec![0, 1, 63, 64]);
    let words = bits.words();
    assert!(word_bit(words, 63) && word_bit(words, 199));
    assert!(!word_bit(words, 2) && !word_bit(words, -1) && !word_bit(words, 1 << 20));
}
