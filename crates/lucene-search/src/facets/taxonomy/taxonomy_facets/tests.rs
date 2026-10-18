//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::taxonomy::FastTaxonomyFacetCounts;
use crate::facets::test_index::{build, collect, doc, ff, match_all};

#[test]
fn ord_values_read_and_write_like_javas_arrays_and_maps() {
    let mut unset: OrdValues<i32> = OrdValues::Unset;
    assert_eq!(unset.get(7).unwrap(), 0);
    unset.set(7, 1).unwrap();
    assert_eq!(unset.get(7).unwrap(), 0, "nothing allocated, nothing kept");

    let mut dense: OrdValues<i32> = OrdValues::new(false, 3);
    dense.set(2, 9).unwrap();
    assert_eq!(dense.get(2).unwrap(), 9);
    for ord in [3, -1] {
        match dense.get(ord) {
            Err(Error::IllegalArgument(m)) => {
                assert_eq!(m, format!("Index {ord} out of bounds for length 3"))
            }
            other => panic!("{other:?}"),
        }
        assert!(dense.set(ord, 1).is_err());
    }

    let mut sparse: OrdValues<f32> = OrdValues::new(true, 3);
    assert_eq!(sparse.get(1000).unwrap(), 0.0);
    sparse.set(1000, 2.5).unwrap();
    assert_eq!(sparse.get(1000).unwrap(), 2.5);
}

#[test]
fn float_compare_is_javas() {
    assert_eq!(java_float_compare(1.0, 2.0), Ordering::Less);
    assert_eq!(java_float_compare(2.0, 1.0), Ordering::Greater);
    assert_eq!(java_float_compare(-0.0, 0.0), Ordering::Less);
    assert_eq!(
        java_float_compare(f32::NAN, f32::INFINITY),
        Ordering::Greater
    );
    assert_eq!(java_float_compare(f32::NAN, -f32::NAN), Ordering::Equal);
    let float = Aggregation::Float(AssociationAggregationFunction::Sum);
    assert_eq!(
        compare_values(float, FacetValue::Int(-1), FacetValue::Float(-0.5)),
        Ordering::Less
    );
    assert_eq!(
        compare_values(Aggregation::Count, FacetValue::Int(2), FacetValue::Int(2)),
        Ordering::Equal
    );
}

#[test]
fn a_missing_component_is_an_error() {
    let l = FacetLabel::new(["a"]).unwrap();
    assert_eq!(component(&l, 0).unwrap(), "a");
    match component(&l, 1) {
        Err(Error::IllegalArgument(m)) => assert_eq!(m, "Index 1 out of bounds for length 1"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn small_taxonomies_count_densely() {
    let config = FacetsConfig::new();
    let index = build(&config, vec![doc("d0", vec![ff("A", &["x"])])], &[], &[]);
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        assert!(!use_hash_table(Some(&fc), 10));
        assert!(!use_hash_table(None, 5000));
        // A big taxonomy, but every document matched.
        assert!(!use_hash_table(Some(&fc), 5000));
        let empty = FacetsCollector::new(false);
        // No segment at all: 0 < 0 is false.
        assert!(!use_hash_table(Some(&empty), 5000));
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        assert!(matches!(facets.counts, OrdValues::Dense(_)));
        assert_eq!(facets.parents().get(2), 1);
    });
}

/// `getTopDims` replaces its worst dim in place but keeps that dim's
/// ordinal, as Java does: a multi-valued dim (whose children are not
/// computed up front) then reports the replaced dim's children.
#[test]
fn top_dims_keeps_the_replaced_dims_ordinal_like_java() {
    let mut config = FacetsConfig::new();
    for d in ["M1", "M2"] {
        config.set_multi_valued(d, true);
        config.set_require_dim_count(d, true);
    }
    let index = build(
        &config,
        vec![
            doc("d0", vec![ff("M1", &["x"])]),
            doc("d1", vec![ff("M1", &["x"])]),
            doc("d2", vec![ff("M1", &["x"])]),
            doc("d3", vec![ff("M2", &["y"])]),
        ],
        &[],
        &[],
    );
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let facets = FastTaxonomyFacetCounts::new(taxo, &config, &fc).unwrap();
        // Root's children are walked newest first: M2 (1) gets in, then M1
        // (3) replaces it.
        let top = facets.get_top_dims(1, 5).unwrap();
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].dim, "M1");
        assert_eq!(top[0].value, FacetValue::Int(1), "M2's dim count");
        assert_eq!(top[0].label_values[0].label, "y", "M2's child");
        // With room for both, each is its own.
        let both = facets.get_top_dims(2, 5).unwrap();
        assert_eq!(
            both.iter()
                .map(|r| (r.dim.as_str(), r.value, r.label_values[0].label.as_str()))
                .collect::<Vec<_>>(),
            vec![
                ("M1", FacetValue::Int(3), "x"),
                ("M2", FacetValue::Int(1), "y")
            ]
        );
    });
}

#[test]
fn dense_slots_refuse_an_ordinal_outside_either_array() {
    let mut values = vec![0.5f32; 3];
    let mut counts = vec![0; 2];
    let (v, c) = dense_slots(&mut values, &mut counts, 1).unwrap();
    *v += 1.0;
    *c += 1;
    assert_eq!((values[1], counts[1]), (1.5, 1));
    for ord in [2, -1] {
        match dense_slots(&mut values, &mut counts, ord) {
            Err(Error::IllegalArgument(m)) => {
                assert_eq!(m, format!("Index {ord} out of bounds for length 2"))
            }
            other => panic!("{other:?}"),
        }
    }
}
