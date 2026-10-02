//! Port of `org.apache.lucene.geo.XYEncodingUtils`.

use super::{java_double_string, java_float_string, GeoError};
use crate::numeric_utils;

/// Port of `org.apache.lucene.geo.XYEncodingUtils`: cartesian values are
/// `float`s indexed as their sortable-int bits.
#[derive(Debug, Clone, Copy)]
pub struct XYEncodingUtils;

impl XYEncodingUtils {
    /// `MIN_VAL_INCL`: `-Float.MAX_VALUE` as a double.
    pub const MIN_VAL_INCL: f64 = -(f32::MAX as f64);
    /// `MAX_VAL_INCL`: `Float.MAX_VALUE` as a double.
    pub const MAX_VAL_INCL: f64 = f32::MAX as f64;

    /// `checkVal`: finite.
    pub fn check_val(x: f32) -> Result<f32, GeoError> {
        if !x.is_finite() {
            return Err(GeoError::illegal(format!(
                "invalid value {}; must be between {} and {}",
                java_float_string(x),
                java_double_string(Self::MIN_VAL_INCL),
                java_double_string(Self::MAX_VAL_INCL)
            )));
        }
        Ok(x)
    }

    /// `encode(float)`.
    pub fn encode(x: f32) -> Result<i32, GeoError> {
        Ok(numeric_utils::float_to_sortable_int(Self::check_val(x)?))
    }

    /// `decode(int)`.
    pub fn decode(encoded: i32) -> f32 {
        numeric_utils::sortable_int_to_float(encoded)
    }

    /// `decode(byte[], int)`.
    pub fn decode_bytes(src: &[u8], offset: usize) -> f32 {
        Self::decode(numeric_utils::sortable_bytes_to_int(src, offset))
    }

    /// `floatArrayToDoubleArray`.
    pub fn float_array_to_double_array(f: &[f32]) -> Vec<f64> {
        f.iter().map(|&v| f64::from(v)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_errors() {
        for v in [0.0f32, -0.0, 1.5, -3.25e30, f32::MAX, f32::MIN_POSITIVE] {
            let e = XYEncodingUtils::encode(v).unwrap();
            assert_eq!(XYEncodingUtils::decode(e).to_bits(), v.to_bits());
            let mut b = [0u8; 4];
            numeric_utils::int_to_sortable_bytes(e, &mut b, 0);
            assert_eq!(XYEncodingUtils::decode_bytes(&b, 0).to_bits(), v.to_bits());
        }
        assert_eq!(
            XYEncodingUtils::encode(f32::NAN).unwrap_err().to_string(),
            "invalid value NaN; must be between -3.4028234663852886E38 and 3.4028234663852886E38"
        );
        assert!(XYEncodingUtils::check_val(f32::NEG_INFINITY).is_err());
        assert_eq!(
            XYEncodingUtils::float_array_to_double_array(&[1.5, 0.1]),
            vec![1.5, f64::from(0.1f32)]
        );
    }
}
