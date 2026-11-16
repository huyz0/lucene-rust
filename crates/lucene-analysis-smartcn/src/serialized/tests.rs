#![cfg(test)]
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use super::test_util::{Value, Writer};
use super::*;

fn read(w: &Writer) -> Result<(ObjectStream, Vec<Option<usize>>), SmartcnError> {
    ObjectStream::read(&w.out)
}

#[test]
fn primitive_and_nested_arrays_round_trip() {
    let mut w = Writer::new();
    w.write(&Value::Short(vec![-1, 2]));
    w.write(&Value::Char(vec![0x4E2D]));
    w.write(&Value::Objects(
        "[[[C",
        vec![
            Value::Null,
            Value::Objects("[[C", vec![Value::Char(vec![1, 2]), Value::Null]),
            Value::Objects("[[C", vec![]),
        ],
    ));
    w.write(&Value::Objects(
        "[[I",
        vec![Value::Int(vec![7]), Value::Null],
    ));
    w.write(&Value::Long(vec![i64::MIN]));
    w.write(&Value::Null);
    let (s, top) = read(&w).unwrap();
    assert_eq!(top.len(), 6);
    assert_eq!(top[5], None);
    assert_eq!(
        s.array_at(top[0].unwrap()),
        Some(("[S", &ArrayData::Short(vec![-1, 2].into())))
    );
    assert_eq!(
        s.array_at(top[1].unwrap()),
        Some(("[C", &ArrayData::Char(vec![0x4E2D].into())))
    );
    let Some((name, ArrayData::Objects(outer))) = s.array_at(top[2].unwrap()) else {
        panic!()
    };
    assert_eq!((name, outer.len(), outer[0]), ("[[[C", 3, None));
    let Some(("[[C", ArrayData::Objects(inner))) = s.array_at(outer[1].unwrap()) else {
        panic!()
    };
    assert_eq!(
        s.array_at(inner[0].unwrap()),
        Some(("[C", &ArrayData::Char(vec![1, 2].into())))
    );
    assert_eq!(
        s.array_at(top[4].unwrap()),
        Some(("[J", &ArrayData::Long(vec![i64::MIN].into())))
    );
    assert_eq!(s.array_at(0), None); // a descriptor
    assert_eq!(s.array_at(10_000), None);
}

#[test]
fn references_to_earlier_arrays() {
    let mut w = Writer::new();
    w.write(&Value::Char(vec![9]));
    // [[C of two elements: the first array by reference, then a null.
    w.out
        .extend_from_slice(&[0x75, 0x72, 0, 3, b'[', b'[', b'C', 0, 0, 0, 0, 0, 0, 0, 0]);
    w.out
        .extend_from_slice(&[2, 0, 0, 0x78, 0x70, 0, 0, 0, 2, 0x71, 0, 0x7E, 0, 1, 0x70]);
    let (s, top) = read(&w).unwrap();
    let Some((_, ArrayData::Objects(e))) = s.array_at(top[1].unwrap()) else {
        panic!()
    };
    assert_eq!(e, &vec![top[0], None]);
}

#[test]
fn malformed_streams_are_errors() {
    let msg = |b: &[u8]| match ObjectStream::read(b) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("accepted {b:02X?}"),
    };
    assert!(msg(&[0xAC, 0xED, 0, 4]).contains("invalid stream header"));
    assert!(msg(&[0xAC]).contains("EOF"));
    assert!(msg(&[0xAC, 0xED, 0, 5, 0x42]).contains("invalid type code: 42"));
    // A reference before any handle; to a descriptor as an object.
    assert!(msg(&[0xAC, 0xED, 0, 5, 0x71, 0, 0x7E, 0, 0]).contains("invalid handle"));
    let mut w = Writer::new();
    w.write(&Value::Int(vec![1]));
    let mut b = w.out.clone();
    b.extend_from_slice(&[0x71, 0, 0x7E, 0, 0]);
    assert!(msg(&b).contains("class descriptor as an object"));
    // An array descriptor referencing an array handle.
    let mut b = w.out.clone();
    b.extend_from_slice(&[0x75, 0x71, 0, 0x7E, 0, 1]);
    assert!(msg(&b).contains("array as a class"));
    assert!(msg(&[0xAC, 0xED, 0, 5, 0x75, 0x42]).contains("descriptor code: 42"));
    // A class that is not a primitive array (the filter's REJECTED).
    let mut w = Writer::new();
    w.write(&Value::Objects("[Ljava.lang.String;", vec![]));
    assert!(msg(&w.out).contains("REJECTED"));
    // A component of the wrong class.
    let mut w = Writer::new();
    w.write(&Value::Objects("[[C", vec![Value::Int(vec![1])]));
    assert!(msg(&w.out).contains("ArrayStoreException"));
    // Lengths: negative, past the end, past the end for objects.
    let mut w = Writer::new();
    w.write(&Value::Int(vec![1, 2]));
    let mut b = w.out.clone();
    let n = b.len();
    b[n - 12..n - 8].copy_from_slice(&(-1i32).to_be_bytes());
    assert!(msg(&b).contains("NegativeArraySize"));
    b[n - 12..n - 8].copy_from_slice(&i32::MAX.to_be_bytes());
    assert!(msg(&b).contains("EOF"));
    let mut w = Writer::new();
    w.write(&Value::Objects("[[C", vec![Value::Null]));
    let mut b = w.out.clone();
    let n = b.len();
    b[n - 5..n - 1].copy_from_slice(&1000i32.to_be_bytes());
    assert!(msg(&b).contains("EOF"));
    // Descriptor shapes Java's array classes never have.
    for (i, (byte, want)) in [
        (1u8, "no fields"),
        (0x77, "annotations"),
        (0x72, "no superclass"),
    ]
    .into_iter()
    .enumerate()
    {
        let mut w = Writer::new();
        w.write(&Value::Int(vec![]));
        let pos = [20, 21, 22][i];
        w.out[pos] = byte;
        assert!(msg(&w.out).contains(want), "{want}: {}", msg(&w.out));
    }
    let mut w = Writer::new();
    w.write(&Value::Int(vec![]));
    w.out[8] = 0xFF;
    assert!(msg(&w.out).contains("UTF-8"));
    // Too deep.
    let mut w = Writer::new();
    w.write(&Value::Objects("[[[[[C", vec![]));
    assert!(msg(&w.out).contains("REJECTED"));
}

#[test]
fn a_self_reference_is_refused() {
    // [[C whose element refers to itself: the wrong component class, so
    // an array can never contain itself.
    let mut b = vec![0xAC, 0xED, 0, 5, 0x75, 0x72, 0, 3, b'[', b'[', b'C'];
    b.extend_from_slice(&[0; 8]);
    b.extend_from_slice(&[2, 0, 0, 0x78, 0x70, 0, 0, 0, 1, 0x71, 0, 0x7E, 0, 1]);
    let e = ObjectStream::read(&b).err().unwrap();
    assert!(e.to_string().contains("ArrayStoreException"), "{e}");
}
