//! Unit tests, after `TestTaxonomyFacetAssociations`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{assoc_config, assoc_index, collect, match_all, term};
use crate::facets::{FacetValue, Facets};
use crate::Error;

fn labels(r: &crate::facets::FacetResult) -> Vec<(String, FacetValue, i32)> {
    r.label_values
        .iter()
        .map(|lv| (lv.label.clone(), lv.value, lv.count))
        .collect()
}

#[test]
fn sums_and_maxes_per_ordinal() {
    let index = assoc_index();
    let config = assoc_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let sum = TaxonomyFacetIntAssociations::with_field(
            "$facets.int",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        let r = sum.get_top_children(10, "int", &[]).unwrap().unwrap();
        // 8 documents with `a` (2 each), 10 with `b` (3 each).
        assert_eq!(
            labels(&r),
            vec![
                ("b".to_string(), FacetValue::Int(30), 10),
                ("a".to_string(), FacetValue::Int(16), 8)
            ]
        );
        assert_eq!(r.value, FacetValue::Int(-1), "multi-valued, no dim count");
        assert_eq!(
            sum.get_specific_value("int", &["a"]).unwrap(),
            FacetValue::Int(16)
        );
        let all = sum.get_all_children("int", &[]).unwrap().unwrap();
        assert_eq!(all.child_count, 2);
        assert_eq!(all.value, FacetValue::Int(-1));

        let max = TaxonomyFacetIntAssociations::with_field(
            "$facets.int",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Max,
        )
        .unwrap();
        let r = max.get_top_children(10, "int", &[]).unwrap().unwrap();
        assert_eq!(
            labels(&r),
            vec![
                ("b".to_string(), FacetValue::Int(3), 10),
                ("a".to_string(), FacetValue::Int(2), 8)
            ]
        );
        assert_eq!(max.get_all_dims(5).unwrap().len(), 1);
        assert_eq!(max.get_top_dims(5, 5).unwrap()[0].dim, "int");
    });
}

#[test]
fn a_query_restricts_the_sums() {
    let index = assoc_index();
    let config = assoc_config();
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "even"), false);
        let sum = TaxonomyFacetIntAssociations::with_field(
            "$facets.int",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        // Even documents: `a` at 0,4,...,28 (8), `b` at 0,6,...,24 (5).
        assert_eq!(
            sum.get_specific_value("int", &["a"]).unwrap(),
            FacetValue::Int(16)
        );
        assert_eq!(
            sum.get_specific_value("int", &["b"]).unwrap(),
            FacetValue::Int(15)
        );
    });
}

#[test]
fn the_default_field_and_a_dim_of_another_field() {
    let index = assoc_index();
    let config = assoc_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        // Nothing is in the default field: no values, yet initialized.
        let def = TaxonomyFacetIntAssociations::new(
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        assert!(def.get_all_dims(10).unwrap().is_empty());
        let sum = TaxonomyFacetIntAssociations::with_field(
            "$facets.int",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        match sum.get_top_children(10, "float", &[]) {
            Err(Error::IllegalArgument(m)) => assert_eq!(
                m,
                "dimension \"float\" cannot be found in field \"$facets.int\", since it was \
                 configured to be indexed into field \"$facets.float\""
            ),
            other => panic!("{other:?}"),
        }
    });
}

#[test]
fn association_bytes_are_big_endian_pairs() {
    let mut seen = Vec::new();
    let bytes = [0, 0, 0, 5, 0xff, 0xff, 0xff, 0xfe, 0, 0, 1, 0, 0, 0, 0, 7];
    for_each_association(&bytes, |ord, v| {
        seen.push((ord, v));
        Ok(())
    })
    .unwrap();
    assert_eq!(seen, vec![(5, -2), (256, 7)]);
    for_each_association(&[], |_, _| panic!("no pair")).unwrap();
    match for_each_association(&bytes[..12], |_, _| Ok(())) {
        Err(Error::IllegalArgument(m)) => {
            assert_eq!(
                m,
                "association bytes of length 12 are not (ordinal, value) pairs"
            )
        }
        other => panic!("{other:?}"),
    }
    // The callback's error stops the walk.
    let r = for_each_association(&bytes, |_, _| Err(Error::IllegalState("stop".into())));
    assert!(matches!(r, Err(Error::IllegalState(_))));
    assert_eq!(be_int(&[0, 0, 0, 1], 1), None);
}

/// 1,200 documents, `int/a{i}` = `i` and `float/f{i}` = 0.5: a taxonomy
/// past 1024 ordinals, so one hit counts sparsely.
fn large_assoc_index() -> crate::facets::test_index::FacetIndex {
    use crate::facets::test_index::{build, doc, float_assoc, int_assoc};
    let docs = (0..1200)
        .map(|i| {
            doc(
                &format!("d{i}"),
                vec![
                    int_assoc(i, "int", &[&format!("a{i}")]),
                    float_assoc(0.5, "float", &[&format!("f{i}")]),
                ],
            )
        })
        .collect();
    build(&assoc_config(), docs, &[], &[])
}

#[test]
fn a_selective_query_over_a_large_taxonomy_sums_sparsely() {
    let index = large_assoc_index();
    let config = assoc_config();
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "d7"), false);
        let sum = TaxonomyFacetIntAssociations::with_field(
            "$facets.int",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        assert!(matches!(sum.int_values, OrdValues::Sparse(_)));
        let r = sum.get_top_children(10, "int", &[]).unwrap().unwrap();
        assert_eq!(labels(&r), vec![("a7".to_string(), FacetValue::Int(7), 1)]);
    });
}
