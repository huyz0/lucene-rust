//! Unit tests.

use super::*;
use crate::facets::facetset::{FloatFacetSet, LongFacetSet};

/// A facet set of no dimensions, which the public sets refuse to build.
struct Empty;

impl FacetSet for Empty {
    fn dims(&self) -> usize {
        0
    }
    fn get_comparable_values(&self) -> Vec<i64> {
        Vec::new()
    }
    fn class_name(&self) -> &str {
        "Empty"
    }
}

#[test]
fn exact_matches_only_the_equal_set() {
    let m = ExactFacetSetMatcher::new("e", &LongFacetSet::new(vec![1, 2]).unwrap()).unwrap();
    assert_eq!((m.label(), m.dims()), ("e", 2));
    assert!(m.matches(&[1, 2]));
    assert!(!m.matches(&[1, 3]));
    let f = ExactFacetSetMatcher::new("f", &FloatFacetSet::new(vec![0.5]).unwrap()).unwrap();
    assert!(f.matches(&[i64::from(float_to_sortable_int(0.5))]));
    assert!(matches!(
        ExactFacetSetMatcher::new("x", &Empty),
        Err(Error::IllegalArgument(m)) if m == "dims must be greater than 0: 0"
    ));
}

#[test]
fn range_matches_inside_its_box_inclusively() {
    let m = RangeFacetSetMatcher::new("r", &[DimRange::new(0, 2), DimRange::new(-5, -5)]).unwrap();
    assert_eq!((m.label(), m.dims()), ("r", 2));
    assert!(m.matches(&[0, -5]));
    assert!(m.matches(&[2, -5]));
    assert!(!m.matches(&[3, -5]));
    assert!(!m.matches(&[1, -4]));
    // A set of more dimensions than the ranges.
    assert!(!m.matches(&[1, -5, 0]));
    assert!(RangeFacetSetMatcher::new("r", &[]).is_err());
}

#[test]
fn dim_ranges_move_exclusive_ends_and_print_as_java() {
    assert_eq!(
        DimRange::from_longs(0, false, 10, false).unwrap(),
        DimRange::new(1, 9)
    );
    assert_eq!(DimRange::new(1, 9).to_string(), "DimRange[min=1, max=9]");
    let d = DimRange::from_doubles(0.0, false, 1.0, false).unwrap();
    assert_eq!(d.min, double_to_sortable_long(f64::from_bits(1)));
    assert_eq!(d.max, double_to_sortable_long(1.0f64.next_down()));
    let f = DimRange::from_floats(-1.0, true, 1.0, true).unwrap();
    assert_eq!(f.max, i64::from(float_to_sortable_int(1.0)));
}
