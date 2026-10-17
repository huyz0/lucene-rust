//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::LabelAndValue;

/// A `Facets` that only answers `getAllDims`, to exercise the defaults.
struct Dims(Vec<&'static str>);

impl Facets for Dims {
    fn get_all_children(&self, _: &str, _: &[&str]) -> Result<Option<FacetResult>> {
        Ok(None)
    }

    fn get_top_children(&self, _: i32, _: &str, _: &[&str]) -> Result<Option<FacetResult>> {
        Ok(None)
    }

    fn get_specific_value(&self, _: &str, _: &[&str]) -> Result<FacetValue> {
        Ok(FacetValue::Int(-1))
    }

    fn get_all_dims(&self, top_n: i32) -> Result<Vec<FacetResult>> {
        validate_top_n(top_n)?;
        Ok(self
            .0
            .iter()
            .map(|d| {
                FacetResult::new(
                    *d,
                    vec![],
                    FacetValue::Int(1),
                    vec![LabelAndValue::new("x", FacetValue::Int(1))],
                    1,
                )
            })
            .collect())
    }
}

#[test]
fn validate_top_n_refuses_non_positive_values() {
    assert!(validate_top_n(1).is_ok());
    for n in [0, -1, i32::MIN] {
        match validate_top_n(n) {
            Err(Error::IllegalArgument(m)) => assert_eq!(m, format!("topN must be > 0 (got: {n})")),
            other => panic!("{other:?}"),
        }
    }
}

#[test]
fn get_top_dims_defaults_to_the_head_of_get_all_dims() {
    let f = Dims(vec!["a", "b", "c"]);
    let dims = |v: Vec<FacetResult>| v.into_iter().map(|r| r.dim).collect::<Vec<_>>();
    assert_eq!(dims(f.get_top_dims(2, 5).unwrap()), vec!["a", "b"]);
    assert_eq!(dims(f.get_top_dims(9, 5).unwrap()), vec!["a", "b", "c"]);
    assert!(f.get_top_dims(0, 5).unwrap().is_empty());
    // `subList(0, -1)`.
    match f.get_top_dims(-1, 5) {
        Err(Error::IllegalArgument(m)) => assert_eq!(m, "fromIndex(0) > toIndex(-1)"),
        other => panic!("{other:?}"),
    }
    // The children count is validated by `getAllDims` itself.
    assert!(f.get_top_dims(2, 0).is_err());
    assert!(f.get_all_children("a", &[]).unwrap().is_none());
    assert!(f.get_top_children(1, "a", &[]).unwrap().is_none());
    assert_eq!(f.get_specific_value("a", &[]).unwrap(), FacetValue::Int(-1));
}
