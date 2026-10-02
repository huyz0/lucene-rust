//! Port of `org.apache.lucene.geo.Circle`.

use super::geo_utils::GeoUtils;
use super::{java_double_string, GeoError};

/// Port of `org.apache.lucene.geo.Circle`: a lat/lon centre and a radius in
/// meters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Circle {
    lat: f64,
    lon: f64,
    radius_meters: f64,
}

impl Circle {
    /// `new Circle(lat, lon, radiusMeters)`.
    pub fn new(lat: f64, lon: f64, radius_meters: f64) -> Result<Circle, GeoError> {
        GeoUtils::check_latitude(lat)?;
        GeoUtils::check_longitude(lon)?;
        if !radius_meters.is_finite() || radius_meters < 0.0 {
            return Err(GeoError::illegal(format!(
                "radiusMeters: '{}' is invalid",
                java_double_string(radius_meters)
            )));
        }
        Ok(Circle {
            lat,
            lon,
            radius_meters,
        })
    }

    /// `getLat()`.
    pub fn lat(&self) -> f64 {
        self.lat
    }

    /// `getLon()`.
    pub fn lon(&self) -> f64 {
        self.lon
    }

    /// `getRadius()`.
    pub fn radius(&self) -> f64 {
        self.radius_meters
    }
}

impl std::fmt::Display for Circle {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Circle([{},{}] radius = {} meters)",
            java_double_string(self.lat),
            java_double_string(self.lon),
            java_double_string(self.radius_meters)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates() {
        let c = Circle::new(1.0, 2.0, 3.0).unwrap();
        assert_eq!((c.lat(), c.lon(), c.radius()), (1.0, 2.0, 3.0));
        assert_eq!(c.to_string(), "Circle([1.0,2.0] radius = 3.0 meters)");
        assert_eq!(
            Circle::new(0.0, 0.0, -1.0).unwrap_err().to_string(),
            "radiusMeters: '-1.0' is invalid"
        );
        assert!(Circle::new(0.0, 0.0, f64::INFINITY).is_err());
        assert!(Circle::new(100.0, 0.0, 1.0).is_err());
        assert!(Circle::new(0.0, 200.0, 1.0).is_err());
    }
}
