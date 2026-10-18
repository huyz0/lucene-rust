//! Unit tests.

use super::*;

#[test]
fn exclusive_ends_move_to_the_next_double() {
    let r = DoubleRange::new("x", 0.0, false, 1.0, false).unwrap();
    assert_eq!(r.min, f64::from_bits(1));
    assert_eq!(r.max, 1.0f64.next_down());
    assert!(r.accept(0.5) && !r.accept(0.0) && !r.accept(1.0) && !r.accept(f64::NAN));
    assert_eq!(Range::label(&r), "x");
    assert_eq!(
        DoubleRange::new("y", -1.5, true, 2.0, true)
            .unwrap()
            .to_string(),
        "DoubleRange(y: -1.5 to 2.0)"
    );
    let l = r.to_long_range();
    assert_eq!(
        (l.min, l.max),
        (
            double_to_sortable_long(r.min),
            double_to_sortable_long(r.max)
        )
    );
}

#[test]
fn nan_and_empty_ranges_are_refused() {
    let msg = |r: Result<DoubleRange>| match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        msg(DoubleRange::new("x", f64::NAN, true, 1.0, true)),
        "min cannot be NaN"
    );
    assert_eq!(
        msg(DoubleRange::new("x", 0.0, true, f64::NAN, true)),
        "max cannot be NaN"
    );
    assert_eq!(
        msg(DoubleRange::new("x", 1.0, false, 1.0, true)),
        "range \"x\" matches nothing"
    );
}
