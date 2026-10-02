//! The `java.lang.Math` operations geo3d uses, with Java's semantics.
//!
//! * `sin`/`cos`/`tan` are `StrictMath`'s fdlibm algorithms
//!   ([`crate::strict_math`]). HotSpot replaces `Math.sin`/`cos`/`tan` with
//!   intrinsic stubs on x86-64 whose result differs from fdlibm in the last
//!   bit for about 3.4% of arguments (measured over 5M uniform arguments in
//!   `[-pi, pi]`), so Lucene's own output is not reproducible across JVMs or
//!   platforms; the fixtures are generated with those intrinsics disabled
//!   (`-XX:DisableIntrinsic=_dsin,_dcos,_dtan`), where `Math.sin` *is*
//!   `StrictMath.sin`, and this port matches them bit for bit. See the
//!   module doc of [`super`] for what that leaves.
//! * `asin`/`acos`/`atan`/`atan2` delegate to `StrictMath` in the JDK and are
//!   never intrinsified, so they are exact everywhere.
//! * `sqrt`, `abs`, `floor` are exact IEEE operations.
//! * `min`/`max` propagate NaN and order `-0.0` below `0.0`; `signum` keeps a
//!   zero's sign and a NaN.

pub(crate) use crate::geo::{java_max as max, java_min as min};
use crate::strict_math;

/// `Math.sin` (as `StrictMath.sin`).
#[inline]
pub(crate) fn sin(x: f64) -> f64 {
    strict_math::sin(x)
}

/// `Math.cos` (as `StrictMath.cos`).
#[inline]
pub(crate) fn cos(x: f64) -> f64 {
    strict_math::cos(x)
}

/// `(Math.sin(x), Math.cos(x))`.
#[inline]
pub(crate) fn sin_cos(x: f64) -> (f64, f64) {
    strict_math::sin_cos(x)
}

/// `Math.tan` (as `StrictMath.tan`).
#[inline]
pub(crate) fn tan(x: f64) -> f64 {
    strict_math::tan(x)
}

/// `Math.asin`.
#[inline]
pub(crate) fn asin(x: f64) -> f64 {
    strict_math::asin(x)
}

/// `Math.acos`.
#[inline]
pub(crate) fn acos(x: f64) -> f64 {
    strict_math::acos(x)
}

/// `Math.atan`.
#[inline]
pub(crate) fn atan(x: f64) -> f64 {
    strict_math::atan(x)
}

/// `Math.atan2`.
#[inline]
pub(crate) fn atan2(y: f64, x: f64) -> f64 {
    strict_math::atan2(y, x)
}

/// `Math.sqrt`.
#[inline]
pub(crate) fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// `Math.floor`.
#[inline]
pub(crate) fn floor(x: f64) -> f64 {
    x.floor()
}

/// `Math.abs`.
#[inline]
pub(crate) fn abs(x: f64) -> f64 {
    x.abs()
}

/// `Math.signum(double)`: `x` itself for a zero or a NaN.
#[inline]
pub(crate) fn signum(x: f64) -> f64 {
    if x == 0.0 || x.is_nan() {
        x
    } else {
        1.0f64.copysign(x)
    }
}

/// `Math.nextUp(double)`.
#[inline]
pub(crate) fn next_up(x: f64) -> f64 {
    x.next_up()
}

/// `Math.nextDown(double)`.
#[inline]
pub(crate) fn next_down(x: f64) -> f64 {
    x.next_down()
}

/// `Math.toRadians`: `angdeg * DEGREES_TO_RADIANS`.
#[inline]
pub fn to_radians(deg: f64) -> f64 {
    deg * 0.017453292519943295
}

/// `Math.toDegrees`: `angrad * RADIANS_TO_DEGREES`.
#[inline]
pub fn to_degrees(rad: f64) -> f64 {
    rad * 57.29577951308232
}

/// `Double.doubleToLongBits`: every NaN collapses to `0x7ff8000000000000`.
#[inline]
pub(crate) fn double_to_long_bits(v: f64) -> i64 {
    if v.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        v.to_bits() as i64
    }
}

/// `Double.hashCode(v)`.
#[inline]
pub(crate) fn double_hash(v: f64) -> i32 {
    let bits = double_to_long_bits(v);
    (bits ^ (bits >> 32)) as i32
}

/// `Double.compare(a, b) == 0`.
#[inline]
pub(crate) fn double_compare_eq(a: f64, b: f64) -> bool {
    double_to_long_bits(a) == double_to_long_bits(b)
}

/// Java's `%` on doubles (`fmod`, the sign of the dividend).
#[inline]
pub(crate) fn rem(a: f64, b: f64) -> f64 {
    a % b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_semantics() {
        assert_eq!(signum(-0.0).to_bits(), (-0.0f64).to_bits());
        assert!(signum(f64::NAN).is_nan());
        assert_eq!(signum(-3.0), -1.0);
        assert_eq!(signum(2.0), 1.0);
        assert_eq!(
            double_to_long_bits(f64::from_bits(0xfff8_0000_0000_0001)),
            0x7ff8_0000_0000_0000
        );
        assert_eq!(double_hash(1.0), 1072693248);
        assert!(double_compare_eq(f64::NAN, f64::NAN));
        assert!(!double_compare_eq(0.0, -0.0));
        assert_eq!(to_degrees(to_radians(90.0)), 90.0);
        assert_eq!(next_up(1.0), 1.0 + f64::EPSILON);
        assert_eq!(next_down(1.0), 1.0 - f64::EPSILON / 2.0);
        assert_eq!(rem(-7.0, 3.0), -1.0);
        assert_eq!(sin_cos(0.0), (0.0, 1.0));
        assert_eq!(max(1.0, 2.0), 2.0);
        assert_eq!(min(1.0, 2.0), 1.0);
        assert_eq!(atan(1.0), std::f64::consts::FRAC_PI_4);
        assert_eq!(atan2(1.0, 1.0), std::f64::consts::FRAC_PI_4);
        assert_eq!(tan(0.0), 0.0);
        assert_eq!(asin(1.0), std::f64::consts::FRAC_PI_2);
        assert_eq!(acos(1.0), 0.0);
        assert_eq!(abs(-1.0), 1.0);
        assert_eq!(sqrt(4.0), 2.0);
    }
}
