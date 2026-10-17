//! Unit tests, after `TestTaxonomyFacetLabels`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{basic_config, basic_docs, build, doc, ff};
use crate::facets::FacetsConfig;
use crate::reader::IndexReader;

fn joined(l: Option<FacetLabel>) -> Option<String> {
    l.map(|l| l.components().join("/"))
}

#[test]
fn labels_of_each_document_in_order() {
    let index = build(&basic_config(), basic_docs(), &[], &[]);
    index.with(|_, reader, taxo| {
        let labels = TaxonomyFacetLabels::new(taxo, "$facets");
        let leaves = IndexReader::leaves(reader);
        assert_eq!(leaves.len(), 1);
        let mut r = labels.get_facet_label_reader(leaves[0]).unwrap();
        let mut doc0 = Vec::new();
        while let Some(l) = joined(r.next_facet_label(0).unwrap()) {
            doc0.push(l);
        }
        doc0.sort();
        assert_eq!(doc0, vec!["Author/Bob", "Publish Date/2010/10/15"]);
        // Skipping documents is fine; going back is not.
        assert_eq!(
            joined(r.next_facet_label_in(2, "Publish Date").unwrap()).as_deref(),
            Some("Publish Date/2012/1/1")
        );
        assert_eq!(r.next_facet_label_in(2, "Publish Date").unwrap(), None);
        match r.next_facet_label(1) {
            Err(Error::IllegalArgument(m)) => {
                assert_eq!(m, "docs out of order: previous docId=2 current docId=1")
            }
            other => panic!("{other:?}"),
        }
        match r.next_facet_label_in(3, "Nope") {
            Err(Error::IllegalArgument(m)) => {
                assert_eq!(m, "Category ordinal not found for facet dimension: Nope")
            }
            other => panic!("{other:?}"),
        }
        // A document of the dim's sibling only: nothing under `Author`
        // remains once `Author` was read.
        assert_eq!(
            joined(r.next_facet_label_in(4, "Author").unwrap()).as_deref(),
            Some("Author/Frank")
        );
        assert_eq!(r.next_facet_label_in(4, "Author").unwrap(), None);
    });
}

#[test]
fn a_document_without_the_field_has_no_label() {
    let config = FacetsConfig::new();
    let index = build(
        &config,
        vec![doc("d0", vec![ff("A", &["x"])]), doc("d1", vec![])],
        &[],
        &[],
    );
    index.with(|_, reader, taxo| {
        let leaf = IndexReader::leaves(reader)[0];
        let mut r = TaxonomyFacetLabels::new(taxo, "$facets")
            .get_facet_label_reader(leaf)
            .unwrap();
        assert_eq!(r.next_facet_label(1).unwrap(), None);
        assert_eq!(r.next_facet_label_in(1, "A").unwrap(), None);
        // A field never indexed reads nothing at all.
        let mut none = TaxonomyFacetLabels::new(taxo, "$nope")
            .get_facet_label_reader(leaf)
            .unwrap();
        assert_eq!(none.next_facet_label(0).unwrap(), None);
    });
}

#[test]
fn an_ordinal_past_the_taxonomy_is_an_error() {
    /// One document (0) holding the one ordinal `self.0`.
    struct One(i64);
    impl crate::reader::DocIdSetIterator for One {
        fn doc_id(&self) -> i32 {
            0
        }
        fn next_doc(&mut self) -> Result<i32> {
            Ok(crate::reader::NO_MORE_DOCS)
        }
        fn advance(&mut self, _: i32) -> Result<i32> {
            Ok(crate::reader::NO_MORE_DOCS)
        }
        fn cost(&self) -> i64 {
            1
        }
    }
    impl crate::reader::DocValuesIterator for One {
        fn advance_exact(&mut self, _: i32) -> Result<bool> {
            Ok(true)
        }
    }
    impl SortedNumericDocValues for One {
        fn next_value(&mut self) -> Result<i64> {
            Ok(self.0)
        }
        fn doc_value_count(&self) -> i32 {
            1
        }
    }
    let config = FacetsConfig::new();
    let index = build(&config, vec![doc("d0", vec![ff("A", &["x"])])], &[], &[]);
    index.with(|_, _, taxo| {
        let size = taxo.get_size().unwrap();
        for bad in [size, -2] {
            let mut r = FacetLabelReader {
                taxo_reader: taxo,
                ordinal_values: Box::new(One(i64::from(bad))),
                current_doc_id: -1,
                current_doc_has_values: false,
                current_pos: 0,
                current_doc_ordinal_count: 0,
            };
            match r.next_facet_label_in(0, "A") {
                Err(Error::IllegalArgument(m)) => {
                    assert_eq!(m, format!("Index {bad} out of bounds for length {size}"))
                }
                other => panic!("{other:?}"),
            }
        }
    });
}
