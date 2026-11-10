//! `java.lang.Math` / `StrictMath` (Painless exposes the `double` forms),
//! on the fdlibm port `StrictMath` uses.

use lucene_util::strict_math as sm;

use crate::api::{double_arg, int_arg};
use crate::value::Value;
use crate::vm::{ApiFn, Throw, Vm};

/// `Math.max`: NaN wins, `0.0 > -0.0`.
pub fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() && b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    a.max(b)
}

/// `Math.min`: NaN wins, `-0.0 < 0.0`.
pub fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        return if a.is_sign_negative() || b.is_sign_negative() {
            -0.0
        } else {
            0.0
        };
    }
    a.min(b)
}

/// `Math.round(double)`: `floor(x + 0.5)` as a saturated `long`.
pub fn round(x: f64) -> i64 {
    if x.is_nan() {
        return 0;
    }
    // Java 7+: exact, without the x + 0.5 double rounding error.
    let f = x.floor();
    let diff = x - f;
    let r = if diff >= 0.5 { f + 1.0 } else { f };
    r as i64
}

/// `Math.rint`: round half to even.
pub fn rint(x: f64) -> f64 {
    x.round_ties_even()
}

fn d1(f: fn(f64) -> f64) -> impl Fn(&[Value]) -> Result<Value, Throw> {
    move |a| Ok(Value::Double(f(double_arg(&a[0])?)))
}

fn signum(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        x
    } else {
        1.0f64.copysign(x)
    }
}

fn ulp(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    let a = x.abs();
    if a == f64::MAX {
        return 2f64.powi(971);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

fn next_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        x
    } else if x == 0.0 {
        f64::from_bits(1)
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() + 1)
    } else {
        f64::from_bits(x.to_bits() - 1)
    }
}

fn next_down(x: f64) -> f64 {
    -next_up(-x)
}

pub fn lookup(name: &str, arity: usize) -> Option<ApiFn> {
    macro_rules! one {
        ($f:expr) => {
            |_vm: &mut Vm, a: &[Value]| -> Result<Value, Throw> {
                Ok(Value::Double(($f)(double_arg(&a[0])?)))
            }
        };
    }
    macro_rules! two {
        ($f:expr) => {
            |_vm: &mut Vm, a: &[Value]| -> Result<Value, Throw> {
                Ok(Value::Double(($f)(double_arg(&a[0])?, double_arg(&a[1])?)))
            }
        };
    }
    let _ = d1;
    Some(match (name, arity) {
        ("abs", 1) => one!(f64::abs),
        ("acos", 1) => one!(sm::acos),
        ("asin", 1) => one!(sm::asin),
        ("atan", 1) => one!(sm::atan),
        ("atan2", 2) => two!(sm::atan2),
        ("cbrt", 1) => one!(f64::cbrt),
        ("ceil", 1) => one!(f64::ceil),
        ("copySign", 2) => two!(f64::copysign),
        ("cos", 1) => one!(sm::cos),
        ("cosh", 1) => one!(sm::cosh),
        ("exp", 1) => one!(sm::exp),
        ("expm1", 1) => one!(sm::expm1),
        ("floor", 1) => one!(f64::floor),
        ("hypot", 2) => two!(sm::hypot),
        ("IEEEremainder", 2) => two!(|x: f64, y: f64| x - (x / y).round_ties_even() * y),
        ("log", 1) => one!(sm::log),
        ("log10", 1) => one!(sm::log10),
        ("log1p", 1) => one!(f64::ln_1p),
        ("max", 2) => two!(java_max),
        ("min", 2) => two!(java_min),
        ("nextAfter", 2) => two!(|x: f64, y: f64| if y > x {
            next_up(x)
        } else if y < x {
            next_down(x)
        } else {
            y
        }),
        ("nextDown", 1) => one!(next_down),
        ("nextUp", 1) => one!(next_up),
        ("pow", 2) => two!(sm::pow),
        ("rint", 1) => one!(rint),
        ("round", 1) => |_vm, a| Ok(Value::Long(round(double_arg(&a[0])?))),
        ("scalb", 2) => |_vm, a| {
            Ok(Value::Double(
                double_arg(&a[0])? * 2f64.powi(int_arg(&a[1])?),
            ))
        },
        ("signum", 1) => one!(signum),
        ("sin", 1) => one!(sm::sin),
        ("sinh", 1) => one!(sm::sinh),
        ("sqrt", 1) => one!(f64::sqrt),
        ("tan", 1) => one!(sm::tan),
        ("tanh", 1) => one!(sm::tanh),
        ("toDegrees", 1) => one!(|x: f64| x * (180.0 / std::f64::consts::PI)),
        ("toRadians", 1) => one!(|x: f64| x * (std::f64::consts::PI / 180.0)),
        ("ulp", 1) => one!(ulp),
        ("random", 0) => |_vm, _a| Ok(Value::Double(0.5)),
        _ => return None,
    })
}
