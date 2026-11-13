//! `GeoAreaFactory` (`org.apache.lucene.spatial3d.geom.GeoAreaFactory`).

use super::geo_bbox_factory::make_geo_bbox;
use super::prelude::*;
use super::shape::GeoAreaObject;
use super::xyz_solid::make_xyz_solid;

/// `makeGeoArea(planetModel, topLat, bottomLat, leftLon, rightLon)`: a box.
pub fn make_geo_area_lat_lon(
    planet_model: &Arc<PlanetModel>,
    top_lat: f64,
    bottom_lat: f64,
    left_lon: f64,
    right_lon: f64,
) -> Result<Arc<dyn GeoAreaObject>> {
    make_geo_bbox(planet_model, top_lat, bottom_lat, left_lon, right_lon).map(|a| a as _)
}

/// `makeGeoArea(planetModel, minX, maxX, minY, maxY, minZ, maxZ)`: an x/y/z
/// solid.
pub fn make_geo_area(
    planet_model: &Arc<PlanetModel>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
) -> Result<Arc<dyn GeoAreaObject>> {
    make_xyz_solid(planet_model, min_x, max_x, min_y, max_y, min_z, max_z).map(|a| a as _)
}
