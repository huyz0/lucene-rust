//! Java's `Math.min` and `Math.max` on doubles, with their `NaN` and signed
//! zero rules.

/// Java's `Double.NaN` (`0x7ff8000000000000L`), spelled out: `f64::NAN`'s
/// bits are not guaranteed.
pub const JAVA_NAN: f64 = f64::from_bits(0x7ff8_0000_0000_0000);

/// Java's `Math.min(double, double)`: `NaN` wins, and `-0.0 < 0.0`.
#[inline]
pub fn java_min(a: f64, b: f64) -> f64 {
    // The ordinary case first: one compare each way decides it.
    if a < b {
        return a;
    }
    if b < a {
        return b;
    }
    java_min_tie(a, b)
}

/// [`java_min`] when neither is below the other: equal (signed zeros), or a
/// `NaN`.
#[cold]
fn java_min_tie(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == 0.0 && b.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    a
}

/// Java's `Math.max(double, double)`.
#[inline]
pub fn java_max(a: f64, b: f64) -> f64 {
    if a > b {
        return a;
    }
    if b > a {
        return b;
    }
    java_max_tie(a, b)
}

/// [`java_max`] when neither is above the other.
#[cold]
fn java_max_tie(a: f64, b: f64) -> f64 {
    if a.is_nan() {
        return a;
    }
    if b.is_nan() {
        return b;
    }
    if a == 0.0 && a.to_bits() == (-0.0f64).to_bits() {
        return b;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_min_and_max_keep_nan_and_signed_zeros() {
        assert!(java_min(f64::NAN, 1.0).is_nan());
        assert!(java_min(1.0, f64::NAN).is_nan());
        assert!(java_max(f64::NAN, 1.0).is_nan());
        assert!(java_max(1.0, f64::NAN).is_nan());
        assert_eq!(java_min(0.0, -0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_min(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
        assert_eq!(java_max(-0.0, 0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_max(0.0, -0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(java_min(2.0, 3.0), 2.0);
        assert_eq!(java_max(2.0, 3.0), 3.0);
    }
}
