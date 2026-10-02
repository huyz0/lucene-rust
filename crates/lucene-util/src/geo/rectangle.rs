//! Port of `org.apache.lucene.geo.Rectangle`.

use super::geo_utils::GeoUtils;
use super::polygon::Polygon;
use super::{java_double_string, java_max, java_min, GeoError};
use crate::{sloppy_math, strict_math};

/// Port of `org.apache.lucene.geo.Rectangle`: a lat/lon box, which crosses
/// the dateline when `max_lon < min_lon`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rectangle {
    /// `minLat`.
    pub min_lat: f64,
    /// `maxLat`.
    pub max_lat: f64,
    /// `minLon`.
    pub min_lon: f64,
    /// `maxLon`.
    pub max_lon: f64,
}

impl Rectangle {
    /// `AXISLAT_ERROR`: the slack `GeoUtils.relate` gives the axis latitude.
    pub const AXISLAT_ERROR: f64 = (0.1 / GeoUtils::EARTH_MEAN_RADIUS_METERS).to_degrees();

    /// `new Rectangle(minLat, maxLat, minLon, maxLon)`.
    pub fn new(
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
    ) -> Result<Rectangle, GeoError> {
        GeoUtils::check_latitude(min_lat)?;
        GeoUtils::check_latitude(max_lat)?;
        GeoUtils::check_longitude(min_lon)?;
        GeoUtils::check_longitude(max_lon)?;
        Ok(Rectangle {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        })
    }

    /// `crossesDateline()`.
    pub fn crosses_dateline(&self) -> bool {
        self.max_lon < self.min_lon
    }

    /// `containsPoint(lat, lon, minLat, maxLat, minLon, maxLon)`.
    #[inline]
    pub fn contains_point(
        lat: f64,
        lon: f64,
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
    ) -> bool {
        lat >= min_lat && lat <= max_lat && lon >= min_lon && lon <= max_lon
    }

    /// `fromPointDistance`: the bounding box of a circle, crossing the
    /// dateline or reaching a pole as needed.
    pub fn from_point_distance(
        center_lat: f64,
        center_lon: f64,
        radius_meters: f64,
    ) -> Result<Rectangle, GeoError> {
        GeoUtils::check_latitude(center_lat)?;
        GeoUtils::check_longitude(center_lon)?;
        let rad_lat = center_lat.to_radians();
        let rad_lon = center_lon.to_radians();
        // LUCENE-7143
        let rad_distance = (radius_meters + 7E-2) / GeoUtils::EARTH_MEAN_RADIUS_METERS;
        let mut min_lat = rad_lat - rad_distance;
        let mut max_lat = rad_lat + rad_distance;
        let mut min_lon;
        let mut max_lon;
        if min_lat > GeoUtils::min_lat_radians() && max_lat < GeoUtils::max_lat_radians() {
            let delta_lon =
                sloppy_math::asin(sloppy_math::sin(rad_distance) / sloppy_math::cos(rad_lat));
            min_lon = rad_lon - delta_lon;
            if min_lon < GeoUtils::min_lon_radians() {
                min_lon += 2.0 * std::f64::consts::PI;
            }
            max_lon = rad_lon + delta_lon;
            if max_lon > GeoUtils::max_lon_radians() {
                max_lon -= 2.0 * std::f64::consts::PI;
            }
        } else {
            // a pole is within the distance
            min_lat = java_max(min_lat, GeoUtils::min_lat_radians());
            max_lat = java_min(max_lat, GeoUtils::max_lat_radians());
            min_lon = GeoUtils::min_lon_radians();
            max_lon = GeoUtils::max_lon_radians();
        }
        Rectangle::new(
            min_lat.to_degrees(),
            max_lat.to_degrees(),
            min_lon.to_degrees(),
            max_lon.to_degrees(),
        )
    }

    /// `axisLat`: the latitude at which a circle touches its bounding box's
    /// meridians.
    pub fn axis_lat(center_lat: f64, radius_meters: f64) -> f64 {
        const PIO2: f64 = std::f64::consts::PI / 2.0;
        let mut l1 = center_lat.to_radians();
        let r = (radius_meters + 7E-2) / GeoUtils::EARTH_MEAN_RADIUS_METERS;
        // if we are within radius range of a pole, the lat is the pole itself
        if l1.abs() + r >= GeoUtils::max_lat_radians() {
            return if center_lat >= 0.0 {
                GeoUtils::MAX_LAT_INCL
            } else {
                GeoUtils::MIN_LAT_INCL
            };
        }
        l1 = if center_lat >= 0.0 {
            PIO2 - l1
        } else {
            l1 + PIO2
        };
        // Java: Math.acos(Math.cos(l1) / Math.cos(r)). `Math.acos` delegates
        // to StrictMath; `Math.cos` is a HotSpot intrinsic (Intel's libm stub
        // on x86-64) that is not StrictMath and is nearly always correctly
        // rounded -- as the platform libm is. Over the fixture corpus libm
        // matches the x86-64 JVM on 1222 of 1223 calls (StrictMath on 1199),
        // the rest within 2 ulps; the fixture test holds the port to that.
        let mut l2 = strict_math::acos(l1.cos() / r.cos());
        l2 = if center_lat >= 0.0 {
            PIO2 - l2
        } else {
            l2 - PIO2
        };
        l2.to_degrees()
    }

    /// `fromPolygon(polygons)`: the bounding box of several polygons.
    pub fn from_polygon(polygons: &[Polygon]) -> Result<Rectangle, GeoError> {
        let mut min_lat = f64::INFINITY;
        let mut max_lat = f64::NEG_INFINITY;
        let mut min_lon = f64::INFINITY;
        let mut max_lon = f64::NEG_INFINITY;
        for p in polygons {
            min_lat = java_min(p.min_lat, min_lat);
            max_lat = java_max(p.max_lat, max_lat);
            min_lon = java_min(p.min_lon, min_lon);
            max_lon = java_max(p.max_lon, max_lon);
        }
        Rectangle::new(min_lat, max_lat, min_lon, max_lon)
    }
}

impl std::fmt::Display for Rectangle {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rectangle(lat={} TO {} lon={} TO {}",
            java_double_string(self.min_lat),
            java_double_string(self.max_lat),
            java_double_string(self.min_lon),
            java_double_string(self.max_lon)
        )?;
        if self.max_lon < self.min_lon {
            f.write_str(" [crosses dateline!]")?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basics() {
        let r = Rectangle::new(-10.0, 10.0, 170.0, -170.0).unwrap();
        assert!(r.crosses_dateline());
        assert_eq!(
            r.to_string(),
            "Rectangle(lat=-10.0 TO 10.0 lon=170.0 TO -170.0 [crosses dateline!])"
        );
        let r = Rectangle::new(-10.0, 10.0, -5.0, 5.0).unwrap();
        assert!(!r.crosses_dateline());
        assert_eq!(
            r.to_string(),
            "Rectangle(lat=-10.0 TO 10.0 lon=-5.0 TO 5.0)"
        );
        assert!(Rectangle::contains_point(0.0, 0.0, -1.0, 1.0, -1.0, 1.0));
        assert!(!Rectangle::contains_point(2.0, 0.0, -1.0, 1.0, -1.0, 1.0));
        assert!(Rectangle::new(-91.0, 0.0, 0.0, 0.0).is_err());
        assert!(Rectangle::new(0.0, 91.0, 0.0, 0.0).is_err());
        assert!(Rectangle::new(0.0, 0.0, -181.0, 0.0).is_err());
        assert!(Rectangle::new(0.0, 0.0, 0.0, 181.0).is_err());
    }

    #[test]
    fn point_distance_boxes() {
        // Near a pole: full longitude range.
        let r = Rectangle::from_point_distance(89.9, 0.0, 100_000.0).unwrap();
        assert_eq!((r.min_lon, r.max_lon, r.max_lat), (-180.0, 180.0, 90.0));
        let r = Rectangle::from_point_distance(-89.9, 0.0, 100_000.0).unwrap();
        assert_eq!(r.min_lat, -90.0);
        // Across the dateline both ways.
        assert!(Rectangle::from_point_distance(0.0, 179.9, 100_000.0)
            .unwrap()
            .crosses_dateline());
        assert!(Rectangle::from_point_distance(0.0, -179.9, 100_000.0)
            .unwrap()
            .crosses_dateline());
        assert!(Rectangle::from_point_distance(95.0, 0.0, 1.0).is_err());
        assert!(Rectangle::from_point_distance(0.0, 195.0, 1.0).is_err());
        assert_eq!(Rectangle::axis_lat(89.0, 500_000.0), 90.0);
        assert_eq!(Rectangle::axis_lat(-89.0, 500_000.0), -90.0);
        let a = Rectangle::axis_lat(45.0, 1000.0);
        assert!(a > 45.0 && a < 45.01, "{a}");
        let a = Rectangle::axis_lat(-45.0, 1000.0);
        assert!(a < -45.0 && a > -45.01, "{a}");
    }

    #[test]
    fn from_polygons() {
        let p1 = Polygon::new(&[0.0, 0.0, 1.0, 0.0], &[0.0, 1.0, 1.0, 0.0], vec![]).unwrap();
        let p2 = Polygon::new(&[-5.0, -5.0, -4.0, -5.0], &[3.0, 4.0, 4.0, 3.0], vec![]).unwrap();
        let r = Rectangle::from_polygon(&[p1, p2]).unwrap();
        assert_eq!(
            (r.min_lat, r.max_lat, r.min_lon, r.max_lon),
            (-5.0, 1.0, 0.0, 4.0)
        );
    }
}
