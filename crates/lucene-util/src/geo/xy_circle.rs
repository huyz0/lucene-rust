//! Port of `org.apache.lucene.geo.XYCircle`.

use super::xy_encoding_utils::XYEncodingUtils;
use super::{java_float_string, GeoError};

/// Port of `org.apache.lucene.geo.XYCircle`: a cartesian circle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XYCircle {
    x: f32,
    y: f32,
    radius: f32,
}

impl XYCircle {
    /// `new XYCircle(x, y, radius)`.
    pub fn new(x: f32, y: f32, radius: f32) -> Result<XYCircle, GeoError> {
        if radius <= 0.0 {
            return Err(GeoError::illegal(format!(
                "radius must be bigger than 0, got {}",
                java_float_string(radius)
            )));
        }
        if !radius.is_finite() {
            return Err(GeoError::illegal(format!(
                "radius must be finite, got {}",
                java_float_string(radius)
            )));
        }
        Ok(XYCircle {
            x: XYEncodingUtils::check_val(x)?,
            y: XYEncodingUtils::check_val(y)?,
            radius,
        })
    }

    /// `getX()`.
    pub fn x(&self) -> f32 {
        self.x
    }

    /// `getY()`.
    pub fn y(&self) -> f32 {
        self.y
    }

    /// `getRadius()`.
    pub fn radius(&self) -> f32 {
        self.radius
    }
}

impl std::fmt::Display for XYCircle {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XYCircle([{},{}] radius = {})",
            java_float_string(self.x),
            java_float_string(self.y),
            java_float_string(self.radius)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates() {
        let c = XYCircle::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!((c.x(), c.y(), c.radius()), (1.0, 2.0, 3.0));
        assert_eq!(c.to_string(), "XYCircle([1.0,2.0] radius = 3.0)");
        assert_eq!(
            XYCircle::new(0.0, 0.0, 0.0).unwrap_err().to_string(),
            "radius must be bigger than 0, got 0.0"
        );
        assert_eq!(
            XYCircle::new(0.0, 0.0, f32::INFINITY)
                .unwrap_err()
                .to_string(),
            "radius must be finite, got Infinity"
        );
        assert!(XYCircle::new(f32::NAN, 0.0, 1.0).is_err());
    }
}
