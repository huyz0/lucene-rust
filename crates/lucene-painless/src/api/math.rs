//! `java.lang.Math` / `StrictMath` (Painless exposes the `double` forms),
//! on the fdlibm port `StrictMath` uses.

use lucene_util::strict_math as sm;

use crate::api::{double_arg, int_arg};
use crate::value::Value;
use crate::vm::ApiFn;

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

macro_rules! methods {
    ($($u:literal => $uf:expr,)* ; $($b:literal => $bf:expr,)*) => {
        /// The `double -> double` methods, for callers that run them unboxed.
        pub fn unary(name: &str) -> Option<fn(f64) -> f64> {
            Some(match name {
                $($u => $uf,)*
                _ => return None,
            })
        }

        /// The `(double, double) -> double` methods.
        pub fn binary(name: &str) -> Option<fn(f64, f64) -> f64> {
            Some(match name {
                $($b => $bf,)*
                _ => return None,
            })
        }

        fn double_method(name: &str, arity: usize) -> Option<ApiFn> {
            Some(match (name, arity) {
                $(($u, 1) => |_vm, a| Ok(Value::Double(($uf)(double_arg(&a[0])?))),)*
                $(($b, 2) => |_vm, a| {
                    Ok(Value::Double(($bf)(double_arg(&a[0])?, double_arg(&a[1])?)))
                },)*
                _ => return None,
            })
        }
    };
}

methods!(
    "abs" => f64::abs,
    "acos" => sm::acos,
    "asin" => sm::asin,
    "atan" => sm::atan,
    "cbrt" => f64::cbrt,
    "ceil" => f64::ceil,
    "cos" => sm::cos,
    "cosh" => sm::cosh,
    "exp" => sm::exp,
    "expm1" => sm::expm1,
    "floor" => f64::floor,
    "log" => sm::log,
    "log10" => sm::log10,
    "log1p" => f64::ln_1p,
    "nextDown" => next_down,
    "nextUp" => next_up,
    "rint" => rint,
    "signum" => signum,
    "sin" => sm::sin,
    "sinh" => sm::sinh,
    "sqrt" => f64::sqrt,
    "tan" => sm::tan,
    "tanh" => sm::tanh,
    "toDegrees" => |x: f64| x * (180.0 / std::f64::consts::PI),
    "toRadians" => |x: f64| x * (std::f64::consts::PI / 180.0),
    "ulp" => ulp,
    ;
    "atan2" => sm::atan2,
    "copySign" => f64::copysign,
    "hypot" => sm::hypot,
    "IEEEremainder" => |x: f64, y: f64| x - (x / y).round_ties_even() * y,
    "max" => java_max,
    "min" => java_min,
    "nextAfter" => next_after,
    "pow" => sm::pow,
);

fn next_after(x: f64, y: f64) -> f64 {
    if y > x {
        next_up(x)
    } else if y < x {
        next_down(x)
    } else {
        y
    }
}

pub fn lookup(name: &str, arity: usize) -> Option<ApiFn> {
    if let Some(f) = double_method(name, arity) {
        return Some(f);
    }
    Some(match (name, arity) {
        ("round", 1) => |_vm, a| Ok(Value::Long(round(double_arg(&a[0])?))),
        ("scalb", 2) => |_vm, a| {
            Ok(Value::Double(
                double_arg(&a[0])? * 2f64.powi(int_arg(&a[1])?),
            ))
        },
        ("random", 0) => |_vm, _a| Ok(Value::Double(0.5)),
        _ => return None,
    })
}
