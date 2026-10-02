//! Port of `org.apache.lucene.geo.XYPoint`.

use super::xy_encoding_utils::XYEncodingUtils;
use super::{java_float_string, GeoError};

/// Port of `org.apache.lucene.geo.XYPoint`: a validated cartesian point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XYPoint {
    x: f32,
    y: f32,
}

impl XYPoint {
    /// `new XYPoint(x, y)`.
    pub fn new(x: f32, y: f32) -> Result<XYPoint, GeoError> {
        let x = XYEncodingUtils::check_val(x)?;
        let y = XYEncodingUtils::check_val(y)?;
        Ok(XYPoint { x, y })
    }

    /// `getX()`.
    pub fn x(&self) -> f32 {
        self.x
    }

    /// `getY()`.
    pub fn y(&self) -> f32 {
        self.y
    }
}

impl std::fmt::Display for XYPoint {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XYPoint({},{})",
            java_float_string(self.x),
            java_float_string(self.y)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_prints() {
        let p = XYPoint::new(1.5, -2.0).unwrap();
        assert_eq!((p.x(), p.y()), (1.5, -2.0));
        assert_eq!(p.to_string(), "XYPoint(1.5,-2.0)");
        assert!(XYPoint::new(f32::NAN, 0.0).is_err());
        assert!(XYPoint::new(0.0, f32::INFINITY).is_err());
    }
}
