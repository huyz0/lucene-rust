//! Unit tests.

use super::*;

#[test]
fn long_ranges_encode_and_print_as_java() {
    let r = LongRange::with_dims("q", vec![-1, 2], vec![3, 4]).unwrap();
    assert_eq!(r.dims(), 2);
    assert_eq!(
        r.to_string(),
        "LongRange(label: q, min: [-1, 2], max: [3, 4])"
    );
    let bytes = r.encode().unwrap();
    assert_eq!(bytes.len(), 32);
    assert_eq!(
        lucene_util::numeric_utils::sortable_bytes_to_long(&bytes, 16),
        3
    );
    // The fields are public: an inverted range built by hand is refused
    // when encoded, as `verifyAndEncode` does.
    let bad = LongRange {
        label: "x".into(),
        min: vec![5],
        max: vec![4],
    };
    assert!(matches!(
        bad.encode(),
        Err(Error::IllegalArgument(m)) if m == "min value (5) is greater than max value (4)"
    ));
}

#[test]
fn double_ranges_encode_and_print_as_java() {
    let r = DoubleRange::new("d", 0.5, true, 1.5, true).unwrap();
    assert_eq!(r.dims(), 1);
    assert_eq!(
        r.to_string(),
        "DoubleRange(label: d, min: [0.5], max: [1.5])"
    );
    assert_eq!(r.encode().unwrap().len(), 16);
    let bad = DoubleRange {
        label: "x".into(),
        min: vec![2.0],
        max: vec![-1.0],
    };
    assert!(matches!(
        bad.encode(),
        Err(Error::IllegalArgument(m)) if m == "min value (2.0) is greater than max value (-1.0)"
    ));
    assert!(DoubleRange::with_dims("x", vec![1.0], vec![0.0]).is_err());
}
