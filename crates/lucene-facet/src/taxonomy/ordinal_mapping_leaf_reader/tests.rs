//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

/// Sorted-numeric values over a fixed list of documents.
struct Docs {
    docs: Vec<(i32, Vec<i64>)>,
    /// The index of the current document; `docs.len()` once exhausted.
    at: usize,
    /// The next value of the current document.
    value: usize,
}

impl Docs {
    fn boxed(docs: Vec<(i32, Vec<i64>)>) -> Box<dyn SortedNumericDocValues + 'static> {
        Box::new(Docs {
            at: usize::MAX,
            docs,
            value: 0,
        })
    }

    fn land(&mut self, at: usize) -> i32 {
        self.at = at.min(self.docs.len());
        self.value = 0;
        self.docs.get(self.at).map_or(NO_MORE_DOCS, |d| d.0)
    }
}

impl DocIdSetIterator for Docs {
    // SENTINEL: `-1` = "not positioned yet", `DocIdSetIterator.docID()`'s
    // contract; the filter under test only advances past it.
    fn doc_id(&self) -> i32 {
        match self.at {
            usize::MAX => -1,
            at => self.docs.get(at).map_or(NO_MORE_DOCS, |d| d.0),
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        let next = if self.at == usize::MAX {
            0
        } else {
            self.at + 1
        };
        Ok(self.land(next))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let at = self
            .docs
            .iter()
            .position(|d| d.0 >= target)
            .unwrap_or(self.docs.len());
        Ok(self.land(at))
    }
    fn cost(&self) -> i64 {
        self.docs.len() as i64
    }
}

impl DocValuesIterator for Docs {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        match self.docs.iter().position(|d| d.0 == target) {
            Some(at) => {
                self.land(at);
                Ok(true)
            }
            None => Ok(false),
        }
    }
}

impl SortedNumericDocValues for Docs {
    fn doc_value_count(&self) -> i32 {
        self.docs.get(self.at).map_or(0, |d| d.1.len() as i32)
    }
    fn next_value(&mut self) -> Result<i64> {
        let v = self.docs[self.at].1[self.value];
        self.value += 1;
        Ok(v)
    }
}

fn filter() -> OrdinalMappingFilter {
    let mut config = FacetsConfig::new();
    config.set_index_field_name("Tag", "$tags");
    // Ordinal i maps to 10 - i: mapping reverses the order.
    let map: Arc<[i32]> = (0..=10).map(|i| 10 - i).collect::<Vec<i32>>().into();
    OrdinalMappingFilter::new(map, &config)
}

fn values(dv: &mut dyn SortedNumericDocValues) -> Vec<i64> {
    (0..dv.doc_value_count())
        .map(|_| dv.next_value().unwrap())
        .collect()
}

#[test]
fn facet_fields_are_every_dims_and_the_default() {
    let f = filter();
    let mut fields: Vec<&str> = f.facet_fields.iter().map(String::as_str).collect();
    fields.sort_unstable();
    assert_eq!(fields, ["$facets", "$tags"]);
    assert!(format!("{f:?}").contains("$tags"));
    // A facet field's values are no longer the codec's singleton.
    assert!(!f.keeps_sorted_numeric_singleton("$facets", true));
    assert!(f.keeps_sorted_numeric_singleton("body", true));
    assert!(!f.keeps_sorted_numeric_singleton("body", false));
}

#[test]
fn other_fields_read_through_unmapped() {
    let f = filter();
    let mut dv = f
        .wrap_sorted_numeric("body", Docs::boxed(vec![(3, vec![1, 2])]))
        .unwrap();
    assert_eq!(dv.next_doc().unwrap(), 3);
    assert_eq!(values(dv.as_mut()), [1, 2]);
}

#[test]
fn facet_ordinals_are_mapped_and_sorted_again() {
    let f = filter();
    let docs = || Docs::boxed(vec![(1, vec![1, 4]), (5, vec![0]), (9, vec![2, 3, 7])]);
    let mut dv = f.wrap_sorted_numeric("$tags", docs()).unwrap();
    assert_eq!(dv.cost(), 3);
    assert_eq!(dv.doc_id(), -1);
    assert_eq!(dv.next_doc().unwrap(), 1);
    assert_eq!(dv.doc_id(), 1);
    assert_eq!(values(dv.as_mut()), [6, 9]);
    // Past the document's values: an error, where Java asserts.
    assert!(matches!(dv.next_value(), Err(Error::IllegalState(_))));
    assert_eq!(dv.advance(6).unwrap(), 9);
    assert_eq!(values(dv.as_mut()), [3, 7, 8]);
    assert_eq!(dv.next_doc().unwrap(), NO_MORE_DOCS);
    assert_eq!(dv.advance(100).unwrap(), NO_MORE_DOCS);

    let mut dv = f.wrap_sorted_numeric("$facets", docs()).unwrap();
    assert!(!dv.advance_exact(4).unwrap());
    assert!(dv.advance_exact(5).unwrap());
    assert_eq!(dv.doc_value_count(), 1);
    assert_eq!(values(dv.as_mut()), [10]);
}

#[test]
fn an_ordinal_outside_the_map_is_refused() {
    let f = filter();
    for bad in [11, -1] {
        let mut dv = f
            .wrap_sorted_numeric("$facets", Docs::boxed(vec![(0, vec![bad])]))
            .unwrap();
        match dv.next_doc() {
            Err(Error::IllegalState(m)) => assert_eq!(
                m,
                format!("ordinal {bad} is outside the ordinal map of 11 ordinals")
            ),
            other => panic!("{other:?}"),
        }
    }
}
