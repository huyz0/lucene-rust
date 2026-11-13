//! `GeoCircleFactory` (`org.apache.lucene.spatial3d.geom.GeoCircleFactory`).

use super::geo_degenerate_point::GeoDegeneratePoint;
use super::geo_exact_circle::GeoExactCircle;
use super::geo_standard_circle::GeoStandardCircle;
use super::prelude::*;
use super::shape::GeoCircle;

/// `makeGeoCircle(planetModel, latitude, longitude, cutoffAngle)`: a point
/// below the angular resolution, a standard circle otherwise.
pub fn make_geo_circle(
    planet_model: &Arc<PlanetModel>,
    latitude: f64,
    longitude: f64,
    cutoff_angle: f64,
) -> Result<Arc<dyn GeoCircle>> {
    if cutoff_angle < MINIMUM_ANGULAR_RESOLUTION {
        return GeoDegeneratePoint::new(planet_model, latitude, longitude)
            .map(|s| Arc::new(s) as _);
    }
    GeoStandardCircle::new(planet_model, latitude, longitude, cutoff_angle)
        .map(|s| Arc::new(s) as _)
}

/// `makeExactGeoCircle(planetModel, latitude, longitude, radius,
/// accuracy)`: a point below the angular resolution, an exact circle
/// otherwise.
pub fn make_exact_geo_circle(
    planet_model: &Arc<PlanetModel>,
    latitude: f64,
    longitude: f64,
    radius: f64,
    accuracy: f64,
) -> Result<Arc<dyn GeoCircle>> {
    if radius < MINIMUM_ANGULAR_RESOLUTION {
        return GeoDegeneratePoint::new(planet_model, latitude, longitude)
            .map(|s| Arc::new(s) as _);
    }
    GeoExactCircle::new(planet_model, latitude, longitude, radius, accuracy)
        .map(|s| Arc::new(s) as _)
}
