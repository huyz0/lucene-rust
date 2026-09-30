//! Port of `org.apache.lucene.util.MathUtil`: integer log, binary gcd, the
//! inverse hyperbolic functions `BM25`-era similarities used, the float
//! summation error bounds `MaxScore` pruning relies on, and `unsignedMin`.
//!
//! `asinh`/`acosh`/`atanh` are Java's formulas over `ln`/`sqrt`, not Rust's
//! `f64::asinh` (a different algorithm with different last-bit rounding).
//! Java's `Math.log` is allowed 1 ulp of error and HotSpot's intrinsic is not
//! glibc's `log`, so these three are within an ulp of Lucene, not bit-exact.

/// `MathUtil.log(long x, int base)`: `floor(log_base(x))`, 0 for `x <= 0`
/// when `base == 2`. `None` for `base <= 1` (Java's `IllegalArgumentException`).
pub fn log_long(x: i64, base: i32) -> Option<i32> {
    if base == 2 {
        return Some(if x <= 0 {
            0
        } else {
            63 - x.leading_zeros() as i32
        });
    }
    if base <= 1 {
        return None;
    }
    let mut x = x;
    let mut ret = 0;
    while x >= base as i64 {
        x /= base as i64;
        ret += 1;
    }
    Some(ret)
}

/// `MathUtil.log(double base, double x)`.
pub fn log_double(base: f64, x: f64) -> f64 {
    x.ln() / base.ln()
}

/// `MathUtil.gcd(a, b)`: binary GCD of the absolute values, with
/// `Long.MIN_VALUE` treated as `2^63` (so `gcd(MIN, 0)` is `MIN`, as in Java).
pub fn gcd(a: i64, b: i64) -> i64 {
    let mut a = a.wrapping_abs();
    let mut b = b.wrapping_abs();
    if a == 0 {
        return b;
    } else if b == 0 {
        return a;
    }
    let common_trailing_zeros = (a | b).trailing_zeros();
    a = ((a as u64) >> a.trailing_zeros()) as i64;
    loop {
        b = ((b as u64) >> b.trailing_zeros()) as i64;
        if a == b {
            break;
        } else if a > b || a == i64::MIN {
            std::mem::swap(&mut a, &mut b);
        }
        if a == 1 {
            break;
        }
        b = b.wrapping_sub(a);
    }
    a.wrapping_shl(common_trailing_zeros)
}

/// `MathUtil.asinh`.
pub fn asinh(a: f64) -> f64 {
    let (a, sign) = if (a.to_bits() as i64) < 0 {
        (a.abs(), -1.0)
    } else {
        (a, 1.0)
    };
    sign * ((a * a + 1.0).sqrt() + a).ln()
}

/// `MathUtil.acosh`.
pub fn acosh(a: f64) -> f64 {
    ((a * a - 1.0).sqrt() + a).ln()
}

/// `MathUtil.atanh`.
pub fn atanh(a: f64) -> f64 {
    let (a, mult) = if (a.to_bits() as i64) < 0 {
        (a.abs(), -0.5)
    } else {
        (a, 0.5)
    };
    mult * ((1.0 + a) / (1.0 - a)).ln()
}

/// `MathUtil.sumRelativeErrorBound(numValues)`: `(n - 1) * 2^-52`, 0 below 2.
pub fn sum_relative_error_bound(num_values: i32) -> f64 {
    if num_values <= 1 {
        return 0.0;
    }
    (num_values - 1) as f64 * f64::EPSILON
}

/// `MathUtil.sumUpperBound(sum, numValues)`.
pub fn sum_upper_bound(sum: f64, num_values: i32) -> f64 {
    if num_values <= 2 {
        return sum;
    }
    let b = sum_relative_error_bound(num_values);
    (1.0 + 2.0 * b) * sum
}

/// `MathUtil.unsignedMin(a, b)`: the smaller of two `int`s read as unsigned.
pub fn unsigned_min(a: i32, b: i32) -> i32 {
    if (a as u32) < (b as u32) {
        a
    } else {
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_log() {
        assert_eq!(log_long(0, 2), Some(0));
        assert_eq!(log_long(-5, 2), Some(0));
        assert_eq!(log_long(1024, 2), Some(10));
        assert_eq!(log_long(1023, 2), Some(9));
        assert_eq!(log_long(1000, 10), Some(3));
        assert_eq!(log_long(999, 10), Some(2));
        assert_eq!(log_long(5, 1), None);
        assert!((log_double(10.0, 1000.0) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn binary_gcd() {
        assert_eq!(gcd(0, 0), 0);
        assert_eq!(gcd(0, 7), 7);
        assert_eq!(gcd(12, 0), 12);
        assert_eq!(gcd(12, 18), 6);
        assert_eq!(gcd(-12, 18), 6);
        assert_eq!(gcd(17, 5), 1);
        assert_eq!(gcd(i64::MIN, 0), i64::MIN);
        assert_eq!(gcd(i64::MIN, 6), 2);
        assert_eq!(gcd(1 << 40, 1 << 20), 1 << 20);
    }

    #[test]
    fn hyperbolic_and_bounds() {
        assert!((asinh(0.5) - 0.5f64.asinh()).abs() < 1e-15);
        assert!((asinh(-2.0) - (-2.0f64).asinh()).abs() < 1e-15);
        assert!((acosh(2.0) - 2.0f64.acosh()).abs() < 1e-15);
        assert!((atanh(0.5) - 0.5f64.atanh()).abs() < 1e-15);
        assert!((atanh(-0.25) - (-0.25f64).atanh()).abs() < 1e-15);
        assert_eq!(sum_relative_error_bound(1), 0.0);
        assert_eq!(sum_relative_error_bound(3), 2.0 * f64::EPSILON);
        assert_eq!(sum_upper_bound(10.0, 2), 10.0);
        assert!(sum_upper_bound(10.0, 5) > 10.0);
        assert_eq!(unsigned_min(-1, 5), 5);
        assert_eq!(unsigned_min(3, 5), 3);
    }
}
