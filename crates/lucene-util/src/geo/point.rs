//! Port of `org.apache.lucene.geo.Point`.

use super::geo_utils::GeoUtils;
use super::{java_double_string, GeoError};

/// Port of `org.apache.lucene.geo.Point`: a validated lat/lon point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    lat: f64,
    lon: f64,
}

impl Point {
    /// `new Point(lat, lon)`.
    pub fn new(lat: f64, lon: f64) -> Result<Point, GeoError> {
        GeoUtils::check_latitude(lat)?;
        GeoUtils::check_longitude(lon)?;
        Ok(Point { lat, lon })
    }

    /// `getLat()`.
    pub fn lat(&self) -> f64 {
        self.lat
    }

    /// `getLon()`.
    pub fn lon(&self) -> f64 {
        self.lon
    }
}

impl std::fmt::Display for Point {
    /// `toString()`: `Point(lon,lat)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Point({},{})",
            java_double_string(self.lon),
            java_double_string(self.lat)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_and_prints() {
        let p = Point::new(10.5, -20.0).unwrap();
        assert_eq!((p.lat(), p.lon()), (10.5, -20.0));
        assert_eq!(p.to_string(), "Point(-20.0,10.5)");
        assert!(Point::new(91.0, 0.0).is_err());
        assert!(Point::new(0.0, 181.0).is_err());
    }
}
