#![cfg(test)]
use super::*;
use crate::query_blob::test_support::Blob;

fn program(function: u8, norm: f32, ops: &[Blob]) -> Blob {
    let mut b = Blob::default().u8(function).f32(norm).i32(ops.len() as i32);
    for op in ops {
        b = b.raw(op);
    }
    b
}

fn op(code: u8) -> Blob {
    Blob::default().u8(code)
}

fn float(x: f32) -> Blob {
    op(0).f32(x)
}

fn double(x: f64) -> Blob {
    op(1).f64(x)
}

fn decoded(b: &Blob) -> std::result::Result<Painless, Status> {
    decode(&mut Cursor::new(&b.0))
}

#[test]
fn float_arithmetic_rounds_as_java_floats() {
    // `1 / (1 + l2Squared(q, v))`: the int literals folded to floats.
    let p = decoded(&program(
        0,
        0.0,
        &[float(1.0), float(1.0), op(2), op(16), op(19)],
    ))
    .unwrap();
    let q = [1.0f32, 2.0];
    let v = [0.5f32, 0.25];
    let d = (1.0f32 - 0.5).powi(2) + (2.0f32 - 0.25).powi(2);
    assert_eq!(p.score_floats(&q, &v).unwrap(), 1.0f32 / (1.0 + d));
    // `1.0 / (1.0 + l2Squared(q, v))`: double arithmetic, then the cast.
    let p = decoded(&program(
        0,
        0.0,
        &[double(1.0), double(1.0), op(2), op(3), op(32), op(35)],
    ))
    .unwrap();
    assert_eq!(
        p.score_floats(&q, &v).unwrap(),
        (1.0f64 / (1.0 + f64::from(d))) as f32
    );
    // Negation and the other operations.
    let p = decoded(&program(
        3,
        0.0,
        &[op(2), op(20), float(3.0), op(17), float(2.0), op(18)],
    ))
    .unwrap();
    let dot = 0.5f32 + 0.5;
    assert_eq!(p.score_floats(&q, &v).unwrap(), (-dot - 3.0) * 2.0);
    let p = decoded(&program(
        3,
        0.0,
        &[
            op(2),
            op(3),
            op(36),
            double(0.5),
            op(33),
            double(4.0),
            op(34),
        ],
    ))
    .unwrap();
    assert_eq!(
        p.score_floats(&q, &v).unwrap(),
        ((-f64::from(dot) - 0.5) * 4.0) as f32
    );
}

#[test]
fn each_function_matches_the_painless_ones() {
    let q = [1.0f32, -2.0, 0.5];
    let v = [0.5f32, 1.0, -1.5];
    let value = |f: u8, norm: f32| {
        decoded(&program(f, norm, &[op(2)]))
            .unwrap()
            .score_floats(&q, &v)
            .unwrap()
    };
    assert_eq!(value(0, 0.0), square_distance(&q, &v));
    assert_eq!(value(1, 0.0), 0.5 + 3.0 + 2.0);
    assert_eq!(value(2, 0.0), 3.0);
    assert_eq!(value(3, 0.0), dot_product(&q, &v));
    let (sum, n1, n2) = cosine_parts_f32(&q, &v);
    assert_eq!(
        value(4, 0.0),
        (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
    );
    let qn: f32 = q.iter().map(|x| x * x).sum();
    let (dot, vn) = q
        .iter()
        .zip(&v)
        .fold((0f32, 0f32), |(d, n), (a, b)| (d + a * b, n + b * b));
    assert_eq!(
        value(5, qn),
        (f64::from(dot) / f64::from(qn * vn).sqrt()) as f32
    );
    assert_eq!(value(5, 0.0), 0.0);

    let qb = [1u8, 0xfe, 3];
    let vb = [2u8, 2, 0xff];
    let bytes = |f: u8| {
        decoded(&program(f, 0.0, &[op(2)]))
            .unwrap()
            .score_bytes(&qb, &vb)
            .unwrap()
    };
    assert_eq!(bytes(0), square_distance_bytes(&qb, &vb) as f32);
    assert_eq!(bytes(1), 1.0 + 4.0 + 4.0);
    assert_eq!(bytes(2), 4.0);
    assert_eq!(bytes(3), dot_product_bytes(&qb, &vb) as f32);
    let (sum, n1, n2) = cosine_sums_i8(&qb, &vb);
    assert_eq!(
        bytes(4),
        (f64::from(sum) / (f64::from(n1) * f64::from(n2)).sqrt()) as f32
    );
}

#[test]
fn what_painless_throws_on_is_an_error() {
    let p = decoded(&program(0, 0.0, &[op(2)])).unwrap();
    assert!(p.score_floats(&[1.0], &[1.0, 2.0]).is_err(), "dimensions");
    assert!(p.score_bytes(&[1], &[1, 2]).is_err(), "dimensions");
    let cos = decoded(&program(4, 0.0, &[op(2)])).unwrap();
    assert!(
        cos.score_floats(&[1.0, 0.0], &[0.0, 0.0]).is_err(),
        "zero vector"
    );
    assert!(cos.score_bytes(&[1, 0], &[0, 0]).is_err(), "zero vector");
    let normed = decoded(&program(5, 1.0, &[op(2)])).unwrap();
    assert!(normed.needs_float_target());
    assert!(normed.score_bytes(&[1], &[1]).is_err());
    assert!(!cos.needs_float_target());
    assert!(missing("v", 3).to_string().contains("'v'"));
    assert!(!cos.describe().is_empty());
}

#[test]
fn bad_programs_are_refused() {
    assert!(decoded(&program(6, 0.0, &[op(2)])).is_err(), "function");
    assert!(decoded(&program(0, 0.0, &[])).is_err(), "empty");
    assert!(
        decoded(&program(0, 0.0, &[float(1.0)])).is_err(),
        "no value"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(2)])).is_err(),
        "two values left"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(16)])).is_err(),
        "pops nothing"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), double(1.0), op(16)])).is_err(),
        "float op on a double"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(32)])).is_err(),
        "double op on floats"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(3), op(3)])).is_err(),
        "widen a double"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(36)])).is_err(),
        "double negate of a float"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(3), op(20)])).is_err(),
        "float negate of a double"
    );
    assert!(
        decoded(&program(0, 0.0, &[op(2), op(99)])).is_err(),
        "operation"
    );
    let deep: Vec<Blob> = std::iter::once(op(2))
        .chain((0..MAX_DEPTH).map(|_| float(1.0)))
        .chain((0..MAX_DEPTH).map(|_| op(16)))
        .collect();
    assert!(decoded(&program(0, 0.0, &deep)).is_err(), "too deep");
    let long: Vec<Blob> = std::iter::once(op(2))
        .chain((0..MAX_OPS).map(|_| op(20)))
        .collect();
    assert!(decoded(&program(0, 0.0, &long)).is_err(), "too long");
    // Truncated.
    assert!(decode(&mut Cursor::new(&[0u8, 0, 0])).is_err());
}
