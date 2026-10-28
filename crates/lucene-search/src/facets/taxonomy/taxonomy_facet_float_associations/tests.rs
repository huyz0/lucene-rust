//! Unit tests, after `TestTaxonomyFacetAssociations` and
//! `TestTaxonomyFacetValueSource`.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{
    assoc_config, assoc_index, basic_config, basic_index, collect, match_all, term,
};
use crate::facets::{FacetValue, Facets};
use crate::values_source;
use crate::Error;

fn float(v: f32) -> FacetValue {
    FacetValue::Float(v)
}

/// `n` additions of `v` in `f32`, as the per-document aggregation sums.
fn summed(v: f32, n: usize) -> f32 {
    (0..n).fold(0f32, |s, _| s + v)
}

#[test]
fn float_associations_sum_and_max() {
    let index = assoc_index();
    let config = assoc_config();
    index.with(|searcher, reader, taxo| {
        let fc = collect(searcher, &match_all(reader), false);
        let sum = TaxonomyFacetFloatAssociations::with_field(
            "$facets.float",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        let r = sum.get_top_children(10, "float", &[]).unwrap().unwrap();
        assert_eq!(r.value, float(-1.0), "multi-valued: the float -1");
        assert_eq!(r.child_count, 2);
        assert_eq!(r.label_values[0].label, "a");
        assert_eq!(r.label_values[0].value, float(summed(0.5, 6)));
        assert_eq!(r.label_values[0].count, 6);
        assert_eq!(r.label_values[1].value, float(summed(0.2, 6)));
        assert_eq!(
            sum.get_specific_value("float", &["b"]).unwrap(),
            float(summed(0.2, 6))
        );
        let all = sum.get_all_children("float", &[]).unwrap().unwrap();
        assert_eq!(all.value, float(-1.0));
        // `getTopDims` ranks a multi-valued dim without a dim count by an
        // `Integer` -1.
        let dims = sum.get_top_dims(2, 1).unwrap();
        assert_eq!(dims.len(), 1);
        assert_eq!(dims[0].value, float(-1.0));

        let max = TaxonomyFacetFloatAssociations::with_field(
            "$facets.float",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Max,
        )
        .unwrap();
        assert_eq!(max.get_specific_value("float", &["a"]).unwrap(), float(0.5));
        assert_eq!(max.get_all_dims(1).unwrap()[0].label_values.len(), 1);
        // The default field has no association values.
        let def = TaxonomyFacetFloatAssociations::new(
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        assert!(def.get_all_dims(1).unwrap().is_empty());
    });
}

#[test]
fn a_values_source_weighs_each_hit_and_rolls_up() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, reader, taxo| {
        let ctx = ValuesContext::new(searcher);
        let fc = collect(searcher, &match_all(reader), false);
        let num = values_source::from_long_field("num");
        let f = TaxonomyFacetFloatAssociations::with_values_source(
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
            num.as_ref(),
            &ctx,
        )
        .unwrap();
        // `num` is 1..=5 in document order.
        let author = f.get_top_children(10, "Author", &[]).unwrap().unwrap();
        assert_eq!(author.value, float(15.0));
        assert_eq!(
            author
                .label_values
                .iter()
                .map(|lv| (lv.label.as_str(), lv.value))
                .collect::<Vec<_>>(),
            vec![
                // A tie goes to the lesser ordinal: Lisa was added first.
                ("Lisa", float(5.0)),
                ("Frank", float(5.0)),
                ("Susan", float(4.0)),
                ("Bob", float(1.0))
            ]
        );
        // The hierarchical dim is rolled up.
        assert_eq!(
            f.get_specific_value("Publish Date", &[]).unwrap(),
            float(15.0)
        );
        assert_eq!(
            f.get_specific_value("Publish Date", &["2012"]).unwrap(),
            float(7.0)
        );
        let dims = f.get_all_dims(1).unwrap();
        assert_eq!(
            dims.iter().map(|r| r.dim.as_str()).collect::<Vec<_>>(),
            vec!["Author", "Publish Date"]
        );
        let top = f.get_top_dims(1, 2).unwrap();
        assert_eq!(top[0].dim, "Author");
    });
}

#[test]
fn kept_scores_feed_the_scores_source() {
    let index = basic_index();
    let config = basic_config();
    index.with(|searcher, reader, taxo| {
        let ctx = ValuesContext::new(searcher);
        let scores = values_source::scores();
        let fc = collect(searcher, &match_all(reader), true);
        let f = TaxonomyFacetFloatAssociations::with_values_source(
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Max,
            scores.as_ref(),
            &ctx,
        )
        .unwrap();
        // `MatchAllDocsQuery` scores every hit 1.
        assert_eq!(
            f.get_specific_value("Author", &["Lisa"]).unwrap(),
            float(1.0)
        );
        // Without kept scores the source has nothing to read.
        let fc = collect(searcher, &term("id", "d2"), false);
        let r = TaxonomyFacetFloatAssociations::with_values_source(
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Max,
            scores.as_ref(),
            &ctx,
        );
        assert!(matches!(r, Err(Error::IllegalArgument(_))));
    });
}

#[test]
fn kept_scores_past_the_last_are_zero() {
    let mut s = KeptScores {
        scores: &[0.5],
        index: -1,
    };
    assert!(s.advance_exact(0).unwrap());
    assert_eq!(s.double_value().unwrap(), 0.5);
    s.advance_exact(3).unwrap();
    assert_eq!(s.double_value().unwrap(), 0.0);
}

#[test]
fn a_selective_query_over_a_large_taxonomy_sums_sparsely() {
    use crate::facets::test_index::{build, doc, float_assoc};
    // 1,200 labels: a taxonomy past 1024 ordinals, so one hit is sparse.
    let docs = (0..1200)
        .map(|i| {
            doc(
                &format!("d{i}"),
                vec![float_assoc(0.5, "float", &[&format!("f{i}")])],
            )
        })
        .collect();
    let config = assoc_config();
    let index = build(&config, docs, &[], &[]);
    index.with(|searcher, _, taxo| {
        let fc = collect(searcher, &term("id", "d7"), false);
        let sum = TaxonomyFacetFloatAssociations::with_field(
            "$facets.float",
            taxo,
            &config,
            &fc,
            AssociationAggregationFunction::Sum,
        )
        .unwrap();
        assert!(matches!(sum.float_values, OrdValues::Sparse(_)));
        let r = sum.get_top_children(10, "float", &[]).unwrap().unwrap();
        assert_eq!(
            r.to_string(),
            "dim=float path=[] value=-1.0 childCount=1\n  f7 (0.5)\n"
        );
    });
}
