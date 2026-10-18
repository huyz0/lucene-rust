//! Unit tests.

use super::*;
use crate::facets::multi_long_values_source::{from_long_field, from_single_valued};
use crate::values_source::long_from_long_field;

#[test]
fn exclusive_ends_move_in_and_empty_ranges_are_refused() {
    let r = LongRange::new("x", 1, false, 4, false).unwrap();
    assert_eq!((r.min, r.max), (2, 3));
    assert_eq!(r.to_string(), "LongRange(x: 2 to 3)");
    assert!(r.accept(2) && r.accept(3) && !r.accept(1) && !r.accept(4));
    assert_eq!(Range::label(&r), "x");
    for bad in [
        LongRange::new("e", i64::MAX, false, i64::MAX, true),
        LongRange::new("e", i64::MIN, true, i64::MIN, false),
        LongRange::new("e", 3, true, 2, true),
    ] {
        assert!(
            matches!(bad, Err(crate::Error::IllegalArgument(m)) if m == "range \"e\" matches nothing")
        );
    }
}

#[test]
fn a_wrapped_single_valued_source_queries_as_one() {
    let r = LongRange::new("q", 0, true, 1, true).unwrap();
    let single = r.get_query_multi(None, from_single_valued(long_from_long_field("f")));
    let multi = r.get_query_multi(None, from_long_field("f"));
    assert_eq!(single.to_string(), multi.to_string());
    assert_ne!(
        format!("{:?}", single.clause()),
        format!("{:?}", multi.clause())
    );
}
