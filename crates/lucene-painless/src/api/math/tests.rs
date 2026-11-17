#![cfg(test)]
use super::*;
use crate::api::test_support::{text, vm_call};

fn un(name: &str, x: f64) -> String {
    text(vm_call(lookup(name, 1).unwrap(), &[Value::Double(x)]))
}

fn bin(name: &str, x: f64, y: f64) -> String {
    text(vm_call(
        lookup(name, 2).unwrap(),
        &[Value::Double(x), Value::Double(y)],
    ))
}

#[test]
fn unary_methods_match_java() {
    let cases: &[(&str, f64, &str)] = &[
        ("abs", -2.5, "2.5"),
        ("ceil", 1.2, "2.0"),
        ("floor", -1.2, "-2.0"),
        ("rint", 2.5, "2.0"),
        ("rint", 3.5, "4.0"),
        ("sqrt", 9.0, "3.0"),
        ("cbrt", 27.0, "3.0"),
        ("signum", -4.0, "-1.0"),
        ("signum", 0.0, "0.0"),
        ("signum", 7.0, "1.0"),
        ("exp", 0.0, "1.0"),
        ("expm1", 0.0, "0.0"),
        ("log", 1.0, "0.0"),
        ("log10", 1000.0, "3.0"),
        ("log1p", 0.0, "0.0"),
        ("sin", 0.0, "0.0"),
        ("cos", 0.0, "1.0"),
        ("tan", 0.0, "0.0"),
        ("sinh", 0.0, "0.0"),
        ("cosh", 0.0, "1.0"),
        ("tanh", 0.0, "0.0"),
        ("asin", 0.0, "0.0"),
        ("acos", 1.0, "0.0"),
        ("atan", 0.0, "0.0"),
        ("toDegrees", std::f64::consts::PI, "180.0"),
        ("toRadians", 180.0, "3.141592653589793"),
        ("ulp", 1.0, "2.220446049250313E-16"),
        ("nextUp", 1.0, "1.0000000000000002"),
        ("nextDown", 1.0, "0.9999999999999999"),
    ];
    for (name, x, want) in cases {
        assert_eq!(un(name, *x), *want, "Math.{name}({x})");
    }
}

#[test]
fn binary_methods_match_java() {
    let cases: &[(&str, f64, f64, &str)] = &[
        ("pow", 2.0, 10.0, "1024.0"),
        ("hypot", 3.0, 4.0, "5.0"),
        ("atan2", 1.0, 1.0, "0.7853981633974483"),
        ("copySign", 3.0, -1.0, "-3.0"),
        ("IEEEremainder", 5.0, 3.0, "-1.0"),
        ("max", 1.0, 2.0, "2.0"),
        ("min", 1.0, 2.0, "1.0"),
        ("nextAfter", 1.0, 2.0, "1.0000000000000002"),
        ("nextAfter", 1.0, 0.0, "0.9999999999999999"),
        ("nextAfter", 1.0, 1.0, "1.0"),
    ];
    for (name, x, y, want) in cases {
        assert_eq!(bin(name, *x, *y), *want, "Math.{name}({x}, {y})");
    }
}

#[test]
fn nan_zero_and_infinity_edges() {
    assert!(java_max(f64::NAN, 1.0).is_nan());
    assert!(java_min(1.0, f64::NAN).is_nan());
    assert!(java_max(-0.0, 0.0).is_sign_positive());
    assert!(java_max(-0.0, -0.0).is_sign_negative());
    assert!(java_min(0.0, -0.0).is_sign_negative());
    assert!(java_min(0.0, 0.0).is_sign_positive());
    assert_eq!(un("nextUp", f64::INFINITY), "Infinity");
    assert_eq!(un("nextUp", 0.0), "4.9E-324");
    assert_eq!(un("nextUp", -1.0), "-0.9999999999999999");
    assert_eq!(un("nextDown", 0.0), "-4.9E-324");
    assert_eq!(un("ulp", f64::NAN), "NaN");
    assert_eq!(un("ulp", f64::NEG_INFINITY), "Infinity");
    assert_eq!(un("ulp", f64::MAX), "1.99584030953472E292");
    assert_eq!(un("signum", f64::NAN), "NaN");
}

#[test]
fn rounding_and_scaling() {
    assert_eq!(round(2.5), 3);
    assert_eq!(round(-2.5), -2);
    assert_eq!(round(-2.6), -3);
    assert_eq!(round(f64::NAN), 0);
    assert_eq!(round(1e30), i64::MAX);
    assert_eq!(round(-1e30), i64::MIN);
    assert_eq!(rint(-2.5), -2.0);
    assert_eq!(
        text(vm_call(
            lookup("round", 1).unwrap(),
            &[Value::Double(0.4999)]
        )),
        "0"
    );
    assert_eq!(
        text(vm_call(
            lookup("scalb", 2).unwrap(),
            &[Value::Double(1.5), Value::Int(3)]
        )),
        "12.0"
    );
    assert_eq!(text(vm_call(lookup("random", 0).unwrap(), &[])), "0.5");
    assert!(lookup("nope", 1).is_none());
    assert!(unary("sqrt").is_some() && unary("nope").is_none());
    assert!(binary("pow").is_some() && binary("nope").is_none());
}
