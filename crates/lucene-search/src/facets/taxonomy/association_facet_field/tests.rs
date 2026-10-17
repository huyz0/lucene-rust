//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;

#[test]
fn int_and_float_fields_encode_big_endian() {
    let f = AssociationFacetField::int(-2, "D", &["a"]).unwrap();
    assert_eq!(f.assoc, vec![0xff, 0xff, 0xff, 0xfe]);
    assert_eq!(f.kind(), AssociationKind::Int);
    assert_eq!(f.dim, "D");
    assert_eq!(f.path, vec!["a"]);
    let g = AssociationFacetField::float(1.0, "D", &["a", "b"]).unwrap();
    assert_eq!(g.assoc, vec![0x3f, 0x80, 0, 0]);
    assert_eq!(g.kind(), AssociationKind::Float);
    // Every `NaN` is the canonical one.
    assert_eq!(
        FloatAssociationFacetField::float_to_bytes_ref(f32::from_bits(0x7fc0_0001)),
        vec![0x7f, 0xc0, 0, 0]
    );
    assert_eq!(
        FloatAssociationFacetField::bytes_ref_to_float(&[0x40, 0x20, 0, 0, 9]),
        Some(2.5)
    );
    assert_eq!(IntAssociationFacetField::bytes_ref_to_int(&[0, 0, 1]), None);
    assert_eq!(FloatAssociationFacetField::bytes_ref_to_float(&[]), None);
    assert_eq!(
        IntAssociationFacetField::bytes_ref_to_int(&IntAssociationFacetField::int_to_bytes_ref(
            i32::MIN
        )),
        Some(i32::MIN)
    );
}

#[test]
fn labels_are_verified_before_the_path_length() {
    let msg = |r: Result<AssociationFacetField>| match r {
        Err(Error::IllegalArgument(m)) => m,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        msg(AssociationFacetField::new::<&str>(vec![1], "D", &[])),
        "path must have at least one element"
    );
    assert!(msg(AssociationFacetField::int(1, "", &["a"])).starts_with("empty or null components"));
    assert!(msg(AssociationFacetField::float(1.0, "D", &["a", ""]))
        .starts_with("empty or null components"));
}

#[test]
fn to_string_is_the_java_classes() {
    assert_eq!(
        AssociationFacetField::int(-7, "A", &["a", "b"])
            .unwrap()
            .to_string(),
        "IntAssociationFacetField(dim=A path=[a, b] value=-7)"
    );
    assert_eq!(
        AssociationFacetField::float(1.5, "A", &["a"])
            .unwrap()
            .to_string(),
        "FloatAssociationFacetField(dim=A path=[a] value=1.5)"
    );
    assert_eq!(
        AssociationFacetField::new(vec![0x0a, 0xff, 0x01], "A", &["a"])
            .unwrap()
            .to_string(),
        "AssociationFacetField(dim=A path=[a] bytes=[a ff 1])"
    );
    assert_eq!(AssociationKind::Bytes.type_name(), "bytes");
    assert_eq!(AssociationKind::Int.type_name(), "int");
    assert_eq!(AssociationKind::Float.type_name(), "float");
}
