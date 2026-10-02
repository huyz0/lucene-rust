//! Port of `org.apache.lucene.geo.XYRectangle`.

use super::xy_encoding_utils::XYEncodingUtils;
use super::{java_float_string, java_max_f32, java_min_f32, GeoError};

/// Port of `org.apache.lucene.geo.XYRectangle`: a cartesian box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct XYRectangle {
    /// `minX`.
    pub min_x: f32,
    /// `maxX`.
    pub max_x: f32,
    /// `minY`.
    pub min_y: f32,
    /// `maxY`.
    pub max_y: f32,
}

impl XYRectangle {
    /// `new XYRectangle(minX, maxX, minY, maxY)`.
    pub fn new(min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> Result<XYRectangle, GeoError> {
        if min_x > max_x {
            return Err(GeoError::illegal(format!(
                "minX must be lower than maxX, got {} > {}",
                java_float_string(min_x),
                java_float_string(max_x)
            )));
        }
        if min_y > max_y {
            return Err(GeoError::illegal(format!(
                "minY must be lower than maxY, got {} > {}",
                java_float_string(min_y),
                java_float_string(max_y)
            )));
        }
        Ok(XYRectangle {
            min_x: XYEncodingUtils::check_val(min_x)?,
            max_x: XYEncodingUtils::check_val(max_x)?,
            min_y: XYEncodingUtils::check_val(min_y)?,
            max_y: XYEncodingUtils::check_val(max_y)?,
        })
    }

    /// `fromPointDistance(x, y, radius)`: the box around a circle, rounded
    /// up one ulp (LUCENE-9243).
    pub fn from_point_distance(x: f32, y: f32, radius: f32) -> Result<XYRectangle, GeoError> {
        XYEncodingUtils::check_val(x)?;
        XYEncodingUtils::check_val(y)?;
        if radius < 0.0 {
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
        let distance_box = radius.next_up();
        let min_x = java_max_f32(-f32::MAX, x - distance_box);
        let max_x = java_min_f32(f32::MAX, x + distance_box);
        let min_y = java_max_f32(-f32::MAX, y - distance_box);
        let max_y = java_min_f32(f32::MAX, y + distance_box);
        XYRectangle::new(min_x, max_x, min_y, max_y)
    }
}

impl std::fmt::Display for XYRectangle {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XYRectangle(x={} TO {} y={} TO {})",
            java_float_string(self.min_x),
            java_float_string(self.max_x),
            java_float_string(self.min_y),
            java_float_string(self.max_y)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let r = XYRectangle::new(-1.0, 1.0, -2.0, 2.0).unwrap();
        assert_eq!(r.to_string(), "XYRectangle(x=-1.0 TO 1.0 y=-2.0 TO 2.0)");
        assert_eq!(
            XYRectangle::new(2.0, 1.0, 0.0, 0.0)
                .unwrap_err()
                .to_string(),
            "minX must be lower than maxX, got 2.0 > 1.0"
        );
        assert_eq!(
            XYRectangle::new(0.0, 0.0, 2.0, 1.0)
                .unwrap_err()
                .to_string(),
            "minY must be lower than maxY, got 2.0 > 1.0"
        );
        assert!(XYRectangle::new(f32::NEG_INFINITY, 0.0, 0.0, 0.0).is_err());
        let r = XYRectangle::from_point_distance(0.0, 0.0, 1.0).unwrap();
        assert_eq!(r.max_x, 1.0f32.next_up());
        let r = XYRectangle::from_point_distance(f32::MAX, -f32::MAX, 1e38).unwrap();
        assert_eq!((r.max_x, r.min_y), (f32::MAX, -f32::MAX));
        assert_eq!(
            XYRectangle::from_point_distance(0.0, 0.0, -1.0)
                .unwrap_err()
                .to_string(),
            "radius must be bigger than 0, got -1.0"
        );
        assert_eq!(
            XYRectangle::from_point_distance(0.0, 0.0, f32::INFINITY)
                .unwrap_err()
                .to_string(),
            "radius must be finite, got Infinity"
        );
        assert!(XYRectangle::from_point_distance(f32::NAN, 0.0, 1.0).is_err());
    }
}
