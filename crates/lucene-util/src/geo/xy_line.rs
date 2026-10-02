//! Port of `org.apache.lucene.geo.XYLine`.

use super::xy_encoding_utils::XYEncodingUtils;
use super::{java_float_string, java_max_f32, java_min_f32, GeoError};

/// Port of `org.apache.lucene.geo.XYLine`: a cartesian polyline.
#[derive(Debug, Clone, PartialEq)]
pub struct XYLine {
    x: Vec<f32>,
    y: Vec<f32>,
    /// `minX`.
    pub min_x: f32,
    /// `maxX`.
    pub max_x: f32,
    /// `minY`.
    pub min_y: f32,
    /// `maxY`.
    pub max_y: f32,
}

impl XYLine {
    /// `new XYLine(x, y)`.
    pub fn new(x: &[f32], y: &[f32]) -> Result<XYLine, GeoError> {
        if x.len() != y.len() {
            return Err(GeoError::illegal("x and y must be equal length"));
        }
        if x.len() < 2 {
            return Err(GeoError::illegal("at least 2 line points required"));
        }
        let mut min_x = f32::MAX;
        let mut min_y = f32::MAX;
        let mut max_x = -f32::MAX;
        let mut max_y = -f32::MAX;
        for i in 0..x.len() {
            min_x = java_min_f32(XYEncodingUtils::check_val(x[i])?, min_x);
            min_y = java_min_f32(XYEncodingUtils::check_val(y[i])?, min_y);
            max_x = java_max_f32(x[i], max_x);
            max_y = java_max_f32(y[i], max_y);
        }
        Ok(XYLine {
            x: x.to_vec(),
            y: y.to_vec(),
            min_x,
            max_x,
            min_y,
            max_y,
        })
    }

    /// `numPoints()`.
    pub fn num_points(&self) -> usize {
        self.x.len()
    }

    /// `getX(vertex)`.
    pub fn x_at(&self, vertex: usize) -> f32 {
        self.x[vertex]
    }

    /// `getY(vertex)`.
    pub fn y_at(&self, vertex: usize) -> f32 {
        self.y[vertex]
    }

    /// `getX()`.
    pub fn x(&self) -> &[f32] {
        &self.x
    }

    /// `getY()`.
    pub fn y(&self) -> &[f32] {
        &self.y
    }
}

impl std::fmt::Display for XYLine {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("XYLine(")?;
        for i in 0..self.x.len() {
            write!(
                f,
                "[{}, {}]",
                java_float_string(self.x[i]),
                java_float_string(self.y[i])
            )?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let l = XYLine::new(&[1.0, -2.0], &[3.0, 4.0]).unwrap();
        assert_eq!((l.min_x, l.max_x, l.min_y, l.max_y), (-2.0, 1.0, 3.0, 4.0));
        assert_eq!(l.num_points(), 2);
        assert_eq!((l.x_at(1), l.y_at(1)), (-2.0, 4.0));
        assert_eq!(l.x(), &[1.0, -2.0]);
        assert_eq!(l.y(), &[3.0, 4.0]);
        assert_eq!(l.to_string(), "XYLine([1.0, 3.0][-2.0, 4.0])");
        assert!(XYLine::new(&[1.0], &[1.0, 2.0]).is_err());
        assert!(XYLine::new(&[1.0], &[1.0]).is_err());
        assert!(XYLine::new(&[1.0, f32::NAN], &[1.0, 2.0]).is_err());
    }
}
