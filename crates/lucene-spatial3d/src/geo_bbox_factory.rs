//! `GeoBBoxFactory` (`org.apache.lucene.spatial3d.geom.GeoBBoxFactory`):
//! the box shape for a latitude/longitude range -- the world, a latitude
//! zone, a longitude slice, a degenerate point or line, or a (wide, north,
//! south) rectangle.

use super::geo_degenerate_horizontal_line::{
    GeoDegenerateHorizontalLine, GeoWideDegenerateHorizontalLine,
};
use super::geo_degenerate_point::GeoDegeneratePoint;
use super::geo_degenerate_vertical_line::{
    GeoDegenerateLatitudeZone, GeoDegenerateLongitudeSlice, GeoDegenerateVerticalLine,
};
use super::geo_latitude_zone::{GeoLatitudeZone, GeoNorthLatitudeZone, GeoSouthLatitudeZone};
use super::geo_longitude_slice::{GeoLongitudeSlice, GeoWideLongitudeSlice};
use super::geo_north_rectangle::GeoNorthRectangle;
use super::geo_rectangle::GeoRectangle;
use super::geo_south_rectangle::GeoSouthRectangle;
use super::geo_wide_north_rectangle::GeoWideNorthRectangle;
use super::geo_wide_rectangle::{GeoWideRectangle, MIN_WIDE_EXTENT};
use super::geo_wide_south_rectangle::GeoWideSouthRectangle;
use super::geo_world::GeoWorld;
use super::lat_lon_bounds::LatLonBounds;
use super::prelude::*;

/// `makeGeoBBox(planetModel, topLat, bottomLat, leftLon, rightLon)`.
pub fn make_geo_bbox(
    planet_model: &Arc<PlanetModel>,
    top_lat: f64,
    bottom_lat: f64,
    left_lon: f64,
    right_lon: f64,
) -> Result<Arc<dyn GeoBBox>> {
    let pm = planet_model;
    let mut top_lat = top_lat;
    let mut bottom_lat = bottom_lat;
    let mut left_lon = left_lon;
    let mut right_lon = right_lon;
    if top_lat > PI * 0.5 {
        top_lat = PI * 0.5;
    }
    if bottom_lat < -PI * 0.5 {
        bottom_lat = -PI * 0.5;
    }
    if left_lon < -PI {
        left_lon = -PI;
    }
    if right_lon > PI {
        right_lon = PI;
    }
    if (longitudes_equals(left_lon, -PI) && longitudes_equals(right_lon, PI))
        || (longitudes_equals(right_lon, -PI) && longitudes_equals(left_lon, PI))
    {
        if is_north_pole(top_lat) && is_south_pole(bottom_lat) {
            return Ok(Arc::new(GeoWorld::new(pm)));
        }
        if latitudes_equals(top_lat, bottom_lat) {
            if is_north_pole(top_lat) {
                return GeoDegeneratePoint::new(pm, top_lat, 0.0).map(|s| Arc::new(s) as _);
            } else if is_south_pole(bottom_lat) {
                return GeoDegeneratePoint::new(pm, bottom_lat, 0.0).map(|s| Arc::new(s) as _);
            }
            return GeoDegenerateLatitudeZone::new(pm, top_lat).map(|s| Arc::new(s) as _);
        }
        if is_north_pole(top_lat) {
            return GeoNorthLatitudeZone::new(pm, bottom_lat).map(|s| Arc::new(s) as _);
        } else if is_south_pole(bottom_lat) {
            return GeoSouthLatitudeZone::new(pm, top_lat).map(|s| Arc::new(s) as _);
        }
        return GeoLatitudeZone::new(pm, top_lat, bottom_lat).map(|s| Arc::new(s) as _);
    }
    let mut extent = right_lon - left_lon;
    if extent < 0.0 {
        extent += PI * 2.0;
    }
    if is_north_pole(top_lat) && is_south_pole(bottom_lat) {
        if longitudes_equals(left_lon, right_lon) {
            return GeoDegenerateLongitudeSlice::new(pm, left_lon).map(|s| Arc::new(s) as _);
        }
        if extent >= PI {
            return GeoWideLongitudeSlice::new(pm, left_lon, right_lon).map(|s| Arc::new(s) as _);
        }
        return GeoLongitudeSlice::new(pm, left_lon, right_lon).map(|s| Arc::new(s) as _);
    }
    if longitudes_equals(left_lon, right_lon) {
        if latitudes_equals(top_lat, bottom_lat) {
            return GeoDegeneratePoint::new(pm, top_lat, left_lon).map(|s| Arc::new(s) as _);
        }
        return GeoDegenerateVerticalLine::new(pm, top_lat, bottom_lat, left_lon)
            .map(|s| Arc::new(s) as _);
    }
    if extent >= MIN_WIDE_EXTENT {
        if latitudes_equals(top_lat, bottom_lat) {
            if is_north_pole(top_lat) {
                return GeoDegeneratePoint::new(pm, top_lat, 0.0).map(|s| Arc::new(s) as _);
            } else if is_south_pole(bottom_lat) {
                return GeoDegeneratePoint::new(pm, bottom_lat, 0.0).map(|s| Arc::new(s) as _);
            }
            return GeoWideDegenerateHorizontalLine::new(pm, top_lat, left_lon, right_lon)
                .map(|s| Arc::new(s) as _);
        }
        if is_north_pole(top_lat) {
            return GeoWideNorthRectangle::new(pm, bottom_lat, left_lon, right_lon)
                .map(|s| Arc::new(s) as _);
        } else if is_south_pole(bottom_lat) {
            return GeoWideSouthRectangle::new(pm, top_lat, left_lon, right_lon)
                .map(|s| Arc::new(s) as _);
        }
        return GeoWideRectangle::new(pm, top_lat, bottom_lat, left_lon, right_lon)
            .map(|s| Arc::new(s) as _);
    }
    if latitudes_equals(top_lat, bottom_lat) {
        if is_north_pole(top_lat) {
            return GeoDegeneratePoint::new(pm, top_lat, 0.0).map(|s| Arc::new(s) as _);
        } else if is_south_pole(bottom_lat) {
            return GeoDegeneratePoint::new(pm, bottom_lat, 0.0).map(|s| Arc::new(s) as _);
        }
        return GeoDegenerateHorizontalLine::new(pm, top_lat, left_lon, right_lon)
            .map(|s| Arc::new(s) as _);
    }
    if is_north_pole(top_lat) {
        return GeoNorthRectangle::new(pm, bottom_lat, left_lon, right_lon)
            .map(|s| Arc::new(s) as _);
    } else if is_south_pole(bottom_lat) {
        return GeoSouthRectangle::new(pm, top_lat, left_lon, right_lon).map(|s| Arc::new(s) as _);
    }
    GeoRectangle::new(pm, top_lat, bottom_lat, left_lon, right_lon).map(|s| Arc::new(s) as _)
}

fn is_north_pole(lat: f64) -> bool {
    latitudes_equals(lat, PI * 0.5)
}

fn is_south_pole(lat: f64) -> bool {
    latitudes_equals(lat, -PI * 0.5)
}

fn latitudes_equals(lat1: f64, lat2: f64) -> bool {
    abs(lat1 - lat2) < MINIMUM_ANGULAR_RESOLUTION || abs(sin(lat1) - sin(lat2)) < MINIMUM_RESOLUTION
}

fn longitudes_equals(lon1: f64, lon2: f64) -> bool {
    abs(lon1 - lon2) < MINIMUM_ANGULAR_RESOLUTION
}

/// `makeGeoBBox(planetModel, LatLonBounds)`: the box for computed bounds.
/// Java unboxes a null latitude or longitude (`NullPointerException`) for
/// bounds nothing was added to; that is an error here.
pub fn make_geo_bbox_from_bounds(
    planet_model: &Arc<PlanetModel>,
    bounds: &LatLonBounds,
) -> Result<Arc<dyn GeoBBox>> {
    let npe = || Error::NullPointer("bounds have no extent".into());
    let top_lat = if bounds.check_no_top_latitude_bound() {
        PI * 0.5
    } else {
        bounds.max_latitude().ok_or_else(npe)?
    };
    let bottom_lat = if bounds.check_no_bottom_latitude_bound() {
        -PI * 0.5
    } else {
        bounds.min_latitude().ok_or_else(npe)?
    };
    let left_lon = if bounds.check_no_longitude_bound() {
        -PI
    } else {
        bounds.left_longitude().ok_or_else(npe)?
    };
    let right_lon = if bounds.check_no_longitude_bound() {
        PI
    } else {
        bounds.right_longitude().ok_or_else(npe)?
    };
    make_geo_bbox(planet_model, top_lat, bottom_lat, left_lon, right_lon)
}
