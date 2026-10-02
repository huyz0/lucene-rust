//! `Tools` (`org.apache.lucene.spatial3d.geom.Tools`).

use super::jmath::acos;

/// `Tools.safeAcos`: `acos` with the argument clamped to `[-1, 1]`, so a value
/// a rounding error past either end does not yield NaN.
pub fn safe_acos(value: f64) -> f64 {
    // `clamp` keeps a NaN as Java's if-chain does.
    acos(value.clamp(-1.0, 1.0))
}

#[cfg(test)]
mod tests {
    #[test]
    fn clamps() {
        assert_eq!(super::safe_acos(1.0000001), 0.0);
        assert_eq!(super::safe_acos(-1.5), std::f64::consts::PI);
        assert!(super::safe_acos(f64::NAN).is_nan());
    }
}
