#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)]
use super::*;

#[test]
fn sortable_encodings_round_trip_and_order() {
    let ints = [i32::MIN, -5, -1, 0, 1, 7, i32::MAX];
    for w in ints.windows(2) {
        assert!(int_to_sortable_bytes(w[0]) < int_to_sortable_bytes(w[1]));
    }
    for v in ints {
        assert_eq!(sortable_bytes_to_int(&int_to_sortable_bytes(v)), v);
    }
    let longs = [i64::MIN, -1, 0, 1, i64::MAX];
    for w in longs.windows(2) {
        assert!(long_to_sortable_bytes(w[0]) < long_to_sortable_bytes(w[1]));
    }
    for v in longs {
        assert_eq!(sortable_bytes_to_long(&long_to_sortable_bytes(v)), v);
    }
    let floats = [f32::NEG_INFINITY, -1.5, -0.0, 0.0, 2.5, f32::INFINITY];
    for w in floats.windows(2) {
        assert!(float_to_sortable_int(w[0]) < float_to_sortable_int(w[1]));
    }
    for v in floats {
        assert_eq!(
            sortable_int_to_float(float_to_sortable_int(v)).to_bits(),
            v.to_bits()
        );
    }
    assert_eq!(
        float_to_sortable_int(f32::NAN),
        float_to_sortable_int(f32::from_bits(0x7fc0_0001))
    );
    let doubles = [f64::NEG_INFINITY, -1.5, -0.0, 0.0, 2.5, f64::INFINITY];
    for w in doubles.windows(2) {
        assert!(double_to_sortable_long(w[0]) < double_to_sortable_long(w[1]));
    }
    for v in doubles {
        assert_eq!(
            sortable_long_to_double(double_to_sortable_long(v)).to_bits(),
            v.to_bits()
        );
    }
    assert_eq!(double_to_sortable_long(f64::NAN), 0x7ff8_0000_0000_0000);
}

#[test]
fn points_pack_and_decode() {
    let p = IntPoint::new("f", &[1, -2]).unwrap();
    assert_eq!(p.packed().len(), 8);
    assert_eq!(IntPoint::decode_dimension(&p.packed()[4..]), -2);
    assert!(p.numeric().is_err(), "two dimensions have no single value");
    assert_eq!(p.numeric_value(), None);
    assert_eq!(p.field_type().point_dimension_count(), 2);
    assert!(IntPoint::new("f", &[]).is_err());
    let mut one = LongPoint::new("l", &[5]).unwrap();
    assert_eq!(one.numeric_value(), Some(Number::Long(5)));
    one.set_values(&[9]).unwrap();
    assert_eq!(one.numeric().unwrap(), Number::Long(9));
    assert!(one.set_values(&[1, 2]).is_err());
    assert_eq!(one.name(), "l");
    assert_eq!(one.binary_value().unwrap().len(), 8);
    assert!(one.token_stream(&Analyzer::keyword()).unwrap().is_none());
    let f = FloatPoint::new("f", &[1.5]).unwrap();
    assert_eq!(f.numeric().unwrap(), Number::Float(1.5));
    let d = DoublePoint::new("d", &[-2.25]).unwrap();
    assert_eq!(d.numeric().unwrap(), Number::Double(-2.25));
    assert!(IntPoint::new("f", &[0; 17]).is_err(), "too many dims");
}

#[test]
fn next_up_and_down_cross_zero_as_lucene_does() {
    assert_eq!(FloatPoint::next_up(-0.0).to_bits(), 0);
    assert_eq!(FloatPoint::next_down(0.0).to_bits(), 0x8000_0000);
    assert_eq!(
        FloatPoint::next_up(1.0),
        f32::from_bits(1.0f32.to_bits() + 1)
    );
    assert_eq!(
        FloatPoint::next_down(1.0),
        f32::from_bits(1.0f32.to_bits() - 1)
    );
    assert_eq!(
        FloatPoint::next_up(-1.0),
        f32::from_bits((-1.0f32).to_bits() - 1)
    );
    assert_eq!(FloatPoint::next_up(f32::INFINITY), f32::INFINITY);
    assert!(FloatPoint::next_up(f32::NAN).is_nan());
    assert_eq!(FloatPoint::next_up(0.0), f32::from_bits(1));
    assert_eq!(DoublePoint::next_up(-0.0).to_bits(), 0);
    assert_eq!(DoublePoint::next_down(0.0).to_bits(), 0x8000_0000_0000_0000);
    assert_eq!(DoublePoint::next_up(0.0), f64::from_bits(1));
    assert_eq!(
        DoublePoint::next_down(1.0),
        f64::from_bits(1.0f64.to_bits() - 1)
    );
    assert_eq!(
        DoublePoint::next_up(-1.0),
        f64::from_bits((-1.0f64).to_bits() - 1)
    );
    assert_eq!(DoublePoint::next_up(f64::INFINITY), f64::INFINITY);
    assert!(DoublePoint::next_down(f64::NAN).is_nan());
}

#[test]
fn binary_points_check_their_dimensions() {
    let p = BinaryPoint::new("b", &[b"ab", b"cd"]).unwrap();
    assert_eq!(p.packed(), b"abcd");
    assert_eq!(p.field_type().point_num_bytes(), 2);
    assert_eq!(p.name(), "b");
    assert_eq!(p.binary_value().unwrap().as_ref(), b"abcd");
    assert!(p.token_stream(&Analyzer::keyword()).unwrap().is_none());
    assert!(BinaryPoint::new("b", &[]).is_err());
    assert!(BinaryPoint::new("b", &[b""]).is_err());
    assert!(BinaryPoint::new("b", &[b"a", b"bc"]).is_err());
    let ft = p.field_type().clone();
    assert!(BinaryPoint::from_packed("b", b"wxyz".to_vec(), ft.clone()).is_ok());
    assert!(BinaryPoint::from_packed("b", b"wxy".to_vec(), ft).is_err());
}

#[test]
fn numeric_fields_index_a_point_and_a_doc_value() {
    let f = IntField::new("i", -3, Store::Yes);
    assert_eq!(
        f.binary_value().unwrap().as_ref(),
        &int_to_sortable_bytes(-3)
    );
    assert_eq!(f.numeric_value(), Some(Number::Int(-3)));
    assert_eq!(f.stored_value(), Some(StoredValue::Int(-3)));
    assert_eq!(f.doc_value(), -3);
    assert!(f.field_type().stored());
    assert_eq!(
        f.field_type().doc_values_type(),
        DocValuesType::SortedNumeric
    );
    let mut f = FloatField::new("f", 1.25, Store::No);
    assert_eq!(f.stored_value(), None);
    assert_eq!(
        f.numeric_value(),
        Some(Number::Long(i64::from(float_to_sortable_int(1.25))))
    );
    f.set_value(2.0);
    assert_eq!(f.value(), 2.0);
    let d = DoubleField::new("d", -0.5, Store::Yes);
    assert_eq!(d.stored_value(), Some(StoredValue::Double(-0.5)));
    assert_eq!(d.doc_value(), double_to_sortable_long(-0.5));
    let l = LongField::new("l", 1 << 40, Store::No);
    assert_eq!(l.binary_value().unwrap().len(), 8);
    assert_eq!(l.name(), "l");
    assert!(l.token_stream(&Analyzer::keyword()).unwrap().is_none());
}

/// `storedValue()` reads back from the sortable doc value, so a NaN
/// payload is stored as Java's canonical NaN; `-0.0` stays `-0.0`.
#[test]
fn numeric_fields_store_canonical_nans() {
    let odd_nan = f32::from_bits(0x7fc0_0001);
    match FloatField::new("f", odd_nan, Store::Yes).stored_value() {
        Some(StoredValue::Float(v)) => assert_eq!(v.to_bits(), 0x7fc0_0000),
        other => panic!("{other:?}"),
    }
    match FloatField::new("f", -0.0, Store::Yes).stored_value() {
        Some(StoredValue::Float(v)) => assert_eq!(v.to_bits(), (-0.0f32).to_bits()),
        other => panic!("{other:?}"),
    }
    let odd_nan = f64::from_bits(0xfff8_0000_0000_0001);
    match DoubleField::new("d", odd_nan, Store::Yes).stored_value() {
        Some(StoredValue::Double(v)) => assert_eq!(v.to_bits(), 0x7ff8_0000_0000_0000),
        other => panic!("{other:?}"),
    }
    assert_eq!(
        LongField::new("l", i64::MIN, Store::Yes).stored_value(),
        Some(StoredValue::Long(i64::MIN))
    );
    assert_eq!(
        IntField::new("i", i32::MIN, Store::Yes).stored_value(),
        Some(StoredValue::Int(i32::MIN))
    );
}

/// `stringValue()` is `fieldsData.toString()`: the doc value as a
/// `Long`, the sortable bits for a float or double field.
#[test]
fn numeric_fields_string_value_is_the_long_doc_value() {
    let s = |f: &dyn IndexableField| f.string_value().map(Cow::into_owned);
    assert_eq!(s(&IntField::new("i", -3, Store::No)).as_deref(), Some("-3"));
    assert_eq!(
        s(&LongField::new("l", i64::MIN, Store::No)).as_deref(),
        Some("-9223372036854775808")
    );
    // `(long) floatToSortableInt(1.25f)` is 1067450368.
    assert_eq!(
        s(&FloatField::new("f", 1.25, Store::No)).as_deref(),
        Some("1067450368")
    );
    assert_eq!(
        s(&DoubleField::new("d", -0.5, Store::No)),
        Some(double_to_sortable_long(-0.5).to_string())
    );
}
