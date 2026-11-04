//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

#[test]
fn number_conversions_follow_java_casts() {
    assert_eq!(FacetValue::Int(-3).int_value(), -3);
    assert_eq!(FacetValue::Float(-2.9).int_value(), -2);
    assert_eq!(FacetValue::Float(f32::NAN).int_value(), 0);
    assert_eq!(FacetValue::Float(1e20).int_value(), i32::MAX);
    assert_eq!(FacetValue::Int(7).float_value(), 7.0);
    assert_eq!(FacetValue::Float(0.5).float_value(), 0.5);
    assert_eq!(FacetValue::Int(7).double_value(), 7.0);
    assert_eq!(FacetValue::Float(0.25).double_value(), 0.25);
    assert_eq!(FacetValue::from(4), FacetValue::Int(4));
    assert_eq!(FacetValue::from(4.0f32), FacetValue::Float(4.0));
}

#[test]
fn equality_is_number_equals() {
    // `Integer.equals(Float)` is false even for the same magnitude.
    assert_ne!(FacetValue::Int(1), FacetValue::Float(1.0));
    // `Float.equals` compares `floatToIntBits`.
    assert_eq!(FacetValue::Float(f32::NAN), FacetValue::Float(-f32::NAN));
    assert_ne!(FacetValue::Float(0.0), FacetValue::Float(-0.0));
}

#[test]
fn hash_codes_match_java() {
    assert_eq!(FacetValue::Int(-5).hash_code(), -5);
    assert_eq!(FacetValue::Float(1.0).hash_code(), 0x3f80_0000);
    assert_eq!(FacetValue::Float(f32::NAN).hash_code(), 0x7fc0_0000);
    // "a".hashCode() == 97.
    let lv = LabelAndValue::new("a", FacetValue::Int(2));
    assert_eq!(lv.hash_code(), 97 + 1439 * 2);
    let r = FacetResult::new("d", vec![], FacetValue::Int(3), vec![lv.clone()], 1);
    assert_eq!(r.hash_code(), lv.hash_code() + 31 * (3 + 31));
    // UTF-16 code units: a supplementary character hashes as its two
    // surrogates.
    assert_eq!(java_string_hash("😀"), 0xd83d * 31 + 0xde00);
}

#[test]
fn label_and_value_counts_and_equality() {
    let a = LabelAndValue::new("x", FacetValue::Float(2.75));
    assert_eq!(a.count, 2, "the two-argument constructor counts intValue()");
    let b = LabelAndValue::with_count("x", FacetValue::Float(2.75), 9);
    assert_eq!(b.count, 9);
    assert_eq!(a, b, "equals ignores the count");
    assert_ne!(a, LabelAndValue::new("y", FacetValue::Float(2.75)));
    assert_eq!(a.to_string(), "x (2.75)");
    assert_eq!(
        LabelAndValue::new("n", FacetValue::Int(3)).to_string(),
        "n (3)"
    );
}

#[test]
fn facet_result_equality_ignores_dim_and_path() {
    let children = vec![LabelAndValue::new("a", FacetValue::Int(1))];
    let a = FacetResult::new(
        "d1",
        vec!["p".into()],
        FacetValue::Int(1),
        children.clone(),
        1,
    );
    let b = FacetResult::new("d2", vec![], FacetValue::Int(1), children.clone(), 1);
    assert_eq!(a, b);
    assert_ne!(
        a,
        FacetResult::new("d1", vec![], FacetValue::Int(1), children.clone(), 2)
    );
    assert_ne!(
        a,
        FacetResult::new("d1", vec![], FacetValue::Int(2), children, 1)
    );
    assert_ne!(
        a,
        FacetResult::new("d1", vec![], FacetValue::Int(1), vec![], 1)
    );
}

#[test]
fn facet_result_to_string_is_javas() {
    let r = FacetResult::new(
        "Publish Date",
        vec!["2001".into(), "3".into()],
        FacetValue::Float(1.5),
        vec![
            LabelAndValue::new("4", FacetValue::Float(1.0)),
            LabelAndValue::new("5", FacetValue::Float(0.5)),
        ],
        2,
    );
    assert_eq!(
        r.to_string(),
        "dim=Publish Date path=[2001, 3] value=1.5 childCount=2\n  4 (1.0)\n  5 (0.5)\n"
    );
}
