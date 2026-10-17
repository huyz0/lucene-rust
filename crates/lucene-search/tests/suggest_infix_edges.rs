//! Boundary and error handling of the infix suggesters (M13 T13.3c) that the
//! Java-fixture differential tests do not reach: the blended constructors and
//! custom coefficient, the writer flow of the blended wrapper, overlapping
//! payloads, and the persistence hooks Java leaves to subclasses.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::suggest::infix::{AnalyzingInfixSuggester, BlendedInfixSuggester, BlenderType};
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;
use lucene_store::{SliceInput, VecDataOutput};

fn plain() -> Arc<Analyzer> {
    Arc::new(Analyzer::standard(None))
}

/// `(text, weight, payload)` rows; `has_payloads` says whether the iterator
/// offers payloads at all.
struct Rows {
    rows: Vec<(&'static str, i64, Option<&'static [u8]>)>,
    has_payloads: bool,
    next: usize,
    current: usize,
}

impl Rows {
    fn new(rows: Vec<(&'static str, i64, Option<&'static [u8]>)>, has_payloads: bool) -> Self {
        Rows {
            rows,
            has_payloads,
            next: 0,
            current: 0,
        }
    }
}

impl InputIterator for Rows {
    fn next(&mut self) -> Option<Vec<u8>> {
        let row = self.rows.get(self.next)?;
        self.current = self.next;
        self.next += 1;
        Some(row.0.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.rows[self.current].1
    }
    fn payload(&self) -> Option<&[u8]> {
        self.rows[self.current].2
    }
    fn has_payloads(&self) -> bool {
        self.has_payloads
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

#[test]
fn a_custom_blender_scores_each_hit_by_its_coefficient() {
    let mut b =
        BlendedInfixSuggester::with_blender(plain(), plain(), 4, BlenderType::Custom, 10, false)
            .unwrap()
            .with_custom_coefficient(|_| 0.5);
    b.build(&mut Rows::new(vec![("red apple pie", 100, None)], false))
        .unwrap();
    let hits = b.lookup("apple", None, false, 1).unwrap();
    assert_eq!(hits.len(), 1);
    // weight 100 * the coefficient 0.5
    assert_eq!(hits[0].value, 50);
    assert_eq!(hits[0].key, "red apple pie");
}

#[test]
fn a_blended_suggester_with_a_position_blender_reads_its_factor_and_exponent() {
    let mut b = BlendedInfixSuggester::with_options(
        plain(),
        plain(),
        4,
        BlenderType::PositionExponentialReciprocal,
        5,
        Some(3.0),
        false,
        true,
        false,
    )
    .unwrap();
    b.build(&mut Rows::new(
        vec![("big red apple", 60, None), ("apple", 60, None)],
        false,
    ))
    .unwrap();
    // "apple" starts with the key and keeps its weight; "apple" is the third
    // token of "big red apple": 60 / (2 + 1)^3 = 2.2, cast to 2
    let hits = b.lookup("apple", None, false, 2).unwrap();
    let scored: Vec<(&str, i64)> = hits.iter().map(|h| (h.key.as_str(), h.value)).collect();
    assert_eq!(scored, [("apple", 60), ("big red apple", 2)]);
    // highlighting was turned off: no highlight key
    assert!(hits.iter().all(|h| h.highlight_key.is_none()));
    // numFactor 5: a lookup for one asks the index for five, answers one
    assert_eq!(b.lookup("apple", None, false, 1).unwrap().len(), 1);
}

#[test]
fn a_blended_suggester_is_a_writer_through_its_infix_suggester() {
    let mut b = BlendedInfixSuggester::new(plain());
    assert!(b.infix().refresh().is_err());
    b.build(&mut Rows::new(vec![("first item", 5, None)], false))
        .unwrap();
    b.infix().add(b"second item", None, 7, None).unwrap();
    assert_eq!(b.lookup("second", None, false, 10).unwrap().len(), 0);
    b.infix().refresh().unwrap();
    assert_eq!(b.lookup("second", None, false, 10).unwrap().len(), 1);
    b.infix().commit().unwrap();
    b.infix().close();
    assert_eq!(b.get_count().unwrap(), 0);
}

#[test]
fn a_blended_suggester_does_not_persist_its_state() {
    let mut b = BlendedInfixSuggester::new(plain());
    b.build(&mut Rows::new(vec![("kept item", 5, None)], false))
        .unwrap();
    assert!(!b.store(&mut VecDataOutput::new()).unwrap());
    assert!(!b.load(&mut SliceInput::new(&[])).unwrap());
    assert_eq!(b.get_count().unwrap(), 1);
}

#[test]
fn an_empty_context_set_filters_nothing() {
    let mut s = AnalyzingInfixSuggester::new(plain());
    s.build(&mut Rows::new(
        vec![("red apple", 5, None), ("blue apple", 4, None)],
        false,
    ))
    .unwrap();
    let all = s.lookup("apple", None, false, 5).unwrap();
    let empty = BTreeSet::new();
    let none_filtered = s.lookup("apple", Some(&empty), false, 5).unwrap();
    assert_eq!(all.len(), 2);
    assert_eq!(all, none_filtered);
}

#[test]
fn equal_hits_with_and_without_payloads_are_both_kept() {
    let mut b = BlendedInfixSuggester::new(plain());
    b.build(&mut Rows::new(
        vec![("dup text", 5, Some(b"a")), ("dup text", 5, None)],
        true,
    ))
    .unwrap();
    // num * numFactor = 200 requested, so both duplicates fit the 2 slots
    let hits = b.lookup("dup", None, false, 20).unwrap();
    assert_eq!(hits.len(), 2);
    // weight 5 is below 10, so it is scaled by 10; the text starts with the key
    assert!(hits.iter().all(|h| h.key == "dup text" && h.value == 50));
}
