//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

fn packed(sets: &[&dyn FacetSet]) -> Vec<u8> {
    FacetSetsField::create("f", sets).unwrap().value().to_vec()
}

#[test]
fn sets_refuse_no_values_and_report_their_dimensions() {
    for e in [
        LongFacetSet::new(vec![]).err(),
        IntFacetSet::new(vec![]).err(),
        FloatFacetSet::new(vec![]).err(),
        DoubleFacetSet::new(vec![]).err(),
    ] {
        assert!(
            matches!(e, Some(Error::IllegalArgument(m)) if m == "values cannot be null or empty")
        );
    }
    let i = IntFacetSet::new(vec![-1, 2]).unwrap();
    assert_eq!((i.dims(), i.size_packed_bytes()), (2, 8));
    assert_eq!(i.get_comparable_values(), vec![-1, 2]);
    let f = FloatFacetSet::new(vec![0.5]).unwrap();
    assert_eq!(f.size_packed_bytes(), 4);
    assert_eq!(
        f.get_comparable_values(),
        vec![i64::from(float_to_sortable_int(0.5))]
    );
    let d = DoubleFacetSet::new(vec![-0.0, 1.0]).unwrap();
    assert_eq!(d.size_packed_bytes(), 16);
    assert_eq!(d.get_comparable_values()[0], double_to_sortable_long(-0.0));
    assert!(d.class_name().ends_with("DoubleFacetSet"));
}

#[test]
fn int_and_float_sets_pack_four_bytes_a_dimension_and_decode_back() {
    let a = IntFacetSet::new(vec![1, -2]).unwrap();
    let b = IntFacetSet::new(vec![i32::MIN, i32::MAX]).unwrap();
    let bytes = packed(&[&a, &b]);
    assert_eq!(bytes.len(), 4 + 2 * 8);
    assert_eq!(sortable_bytes_to_int(&bytes, 0), 2);
    let mut dest = [0i64; 2];
    assert_eq!(decode_ints(&bytes, 4, &mut dest).unwrap(), 8);
    assert_eq!(dest, [1, -2]);
    assert_eq!(decode_ints(&bytes, 12, &mut dest).unwrap(), 8);
    assert_eq!(dest, [i64::from(i32::MIN), i64::from(i32::MAX)]);

    let f = FloatFacetSet::new(vec![0.5, -1.5]).unwrap();
    let bytes = packed(&[&f]);
    assert_eq!(bytes.len(), 12);
    decode_ints(&bytes, 4, &mut dest).unwrap();
    assert_eq!(dest.to_vec(), f.get_comparable_values());
}

#[test]
fn long_and_double_sets_pack_eight_bytes_a_dimension() {
    let l = LongFacetSet::new(vec![i64::MIN, 7]).unwrap();
    let bytes = packed(&[&l, &l]);
    assert_eq!(bytes.len(), 4 + 2 * 16);
    let mut dest = [0i64; 2];
    assert_eq!(decode_longs(&bytes, 20, &mut dest).unwrap(), 16);
    assert_eq!(dest, [i64::MIN, 7]);
    let d = DoubleFacetSet::new(vec![2.5]).unwrap();
    let bytes = packed(&[&d]);
    let mut one = [0i64; 1];
    decode_longs(&bytes, 4, &mut one).unwrap();
    assert_eq!(one[0], double_to_sortable_long(2.5));
}

#[test]
fn create_refuses_no_sets_and_mixed_dimensions_or_types() {
    let msg = |r: Result<BinaryDocValuesField>| match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        msg(FacetSetsField::create("f", &[])),
        "FacetSets cannot be null or empty!"
    );
    let l2 = LongFacetSet::new(vec![1, 2]).unwrap();
    let l1 = LongFacetSet::new(vec![1]).unwrap();
    let i1 = IntFacetSet::new(vec![1]).unwrap();
    assert_eq!(
        msg(FacetSetsField::create("f", &[&l2, &l1])),
        "All FacetSets must have the same number of dimensions. Expected 2 found 1"
    );
    assert!(msg(FacetSetsField::create("f", &[&l1, &i1])).ends_with("IntFacetSet"));
}

#[test]
fn decoders_refuse_a_truncated_value() {
    let bytes = [0u8; 10];
    let mut dest = [0i64; 2];
    for r in [
        decode_longs(&bytes, 0, &mut dest),
        decode_ints(&bytes, 4, &mut dest),
        decode_longs(&bytes, usize::MAX, &mut dest),
    ] {
        assert!(
            matches!(r, Err(Error::IllegalState(m)) if m.contains("runs past the 10-byte value"))
        );
    }
}
