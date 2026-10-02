//! `DistanceUtils`, `DistanceCalculator` and its implementations
//! (`GeodesicSphereDistCalc`'s haversine, law of cosines and Vincenty;
//! `CartesianDistCalc`) from `org.locationtech.spatial4j.distance`.

use std::any::Any;
use std::f64::consts::PI;
use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::shape::{Circle, Point, Rectangle};
use super::Result;
use crate::strict_math::{acos, asin, atan2, cos, sin};

/// `DistanceUtils`: distance constants and the spherical formulas.
#[derive(Debug, Clone, Copy)]
pub struct DistanceUtils;

impl DistanceUtils {
    pub const DEG_90_AS_RADS: f64 = PI / 2.0;
    pub const DEG_180_AS_RADS: f64 = PI;
    pub const DEGREES_TO_RADIANS: f64 = PI / 180.0;
    pub const RADIANS_TO_DEGREES: f64 = 1.0 / Self::DEGREES_TO_RADIANS;
    pub const KM_TO_MILES: f64 = 0.621371192;
    pub const MILES_TO_KM: f64 = 1.0 / Self::KM_TO_MILES;
    /// The IUGG's mean Earth radius in kilometres.
    pub const EARTH_MEAN_RADIUS_KM: f64 = 6371.0087714;
    pub const EARTH_EQUATORIAL_RADIUS_KM: f64 = 6378.1370;
    /// `degrees2Dist(1, EARTH_MEAN_RADIUS_KM)`.
    pub const DEG_TO_KM: f64 = Self::DEGREES_TO_RADIANS * Self::EARTH_MEAN_RADIUS_KM;
    pub const KM_TO_DEG: f64 = 1.0 / Self::DEG_TO_KM;
    pub const EARTH_MEAN_RADIUS_MI: f64 = Self::EARTH_MEAN_RADIUS_KM * Self::KM_TO_MILES;
    pub const EARTH_EQUATORIAL_RADIUS_MI: f64 =
        Self::EARTH_EQUATORIAL_RADIUS_KM * Self::KM_TO_MILES;

    /// `pointOnBearingRAD(startLat, startLon, distanceRAD, bearingRAD, ..)`
    /// before the point is made: the destination's `(lon, lat)` in radians.
    pub fn point_on_bearing_rad(
        start_lat: f64,
        start_lon: f64,
        distance_rad: f64,
        bearing_rad: f64,
    ) -> (f64, f64) {
        let cos_ang_dist = cos(distance_rad);
        let cos_start_lat = cos(start_lat);
        let sin_ang_dist = sin(distance_rad);
        let sin_start_lat = sin(start_lat);
        let sin_lat2 =
            sin_start_lat * cos_ang_dist + cos_start_lat * sin_ang_dist * cos(bearing_rad);
        let lat2 = asin(sin_lat2);
        let lon2 = start_lon
            + atan2(
                sin(bearing_rad) * sin_ang_dist * cos_start_lat,
                cos_ang_dist - sin_start_lat * sin_lat2,
            );
        Self::normalize_lon_lat_rad(lon2, lat2)
    }

    /// `pointOnBearingRAD`'s normalisation of the destination: the
    /// longitude into `[-PI, PI]`, then a latitude past a pole reflected back
    /// (unreachable from `asin`'s range, kept as Java has it).
    fn normalize_lon_lat_rad(lon2: f64, lat2: f64) -> (f64, f64) {
        let (mut lon2, mut lat2) = (lon2, lat2);
        // normalize lon first
        if lon2 > Self::DEG_180_AS_RADS {
            lon2 = -(Self::DEG_180_AS_RADS - (lon2 - Self::DEG_180_AS_RADS));
        } else if lon2 < -Self::DEG_180_AS_RADS {
            lon2 = (lon2 + Self::DEG_180_AS_RADS) + Self::DEG_180_AS_RADS;
        }
        // normalize lat - could flip poles
        if lat2 > Self::DEG_90_AS_RADS {
            lat2 = Self::DEG_90_AS_RADS - (lat2 - Self::DEG_90_AS_RADS);
            if lon2 < 0.0 {
                lon2 += Self::DEG_180_AS_RADS;
            } else {
                lon2 -= Self::DEG_180_AS_RADS;
            }
        } else if lat2 < -Self::DEG_90_AS_RADS {
            lat2 = -Self::DEG_90_AS_RADS - (lat2 + Self::DEG_90_AS_RADS);
            if lon2 < 0.0 {
                lon2 += Self::DEG_180_AS_RADS;
            } else {
                lon2 -= Self::DEG_180_AS_RADS;
            }
        }
        (lon2, lat2)
    }

    /// `normLonDEG(lon)`: into `[-180, 180]`.
    pub fn norm_lon_deg(lon_deg: f64) -> f64 {
        if lon_deg >= -180.0 && lon_deg <= 180.0 {
            return lon_deg;
        }
        let off = (lon_deg + 180.0) % 360.0;
        if off < 0.0 {
            180.0 + off
        } else if off == 0.0 && lon_deg > 0.0 {
            180.0
        } else {
            -180.0 + off
        }
    }

    /// `normLatDEG(lat)`: into `[-90, 90]`, reflecting over the poles.
    pub fn norm_lat_deg(lat_deg: f64) -> f64 {
        if lat_deg >= -90.0 && lat_deg <= 90.0 {
            return lat_deg;
        }
        let off = ((lat_deg + 90.0) % 360.0).abs();
        (if off <= 180.0 { off } else { 360.0 - off }) - 90.0
    }

    /// `calcBoxByDistFromPtDEG(lat, lon, distDEG, ctx, null)` before the
    /// rectangle is made: `[minX, maxX, minY, maxY]`.
    pub fn calc_box_by_dist_from_pt_deg(lat: f64, lon: f64, dist_deg: f64) -> [f64; 4] {
        let (min_x, max_x, mut min_y, mut max_y);
        if dist_deg == 0.0 {
            min_x = lon;
            max_x = lon;
            min_y = lat;
            max_y = lat;
        } else if dist_deg >= 180.0 {
            min_x = -180.0;
            max_x = 180.0;
            min_y = -90.0;
            max_y = 90.0;
        } else {
            max_y = lat + dist_deg;
            min_y = lat - dist_deg;
            if max_y >= 90.0 || min_y <= -90.0 {
                if max_y <= 90.0 && min_y >= -90.0 {
                    // doesn't pass either pole: 180 deg
                    min_x = Self::norm_lon_deg(lon - 90.0);
                    max_x = Self::norm_lon_deg(lon + 90.0);
                } else {
                    min_x = -180.0;
                    max_x = 180.0;
                }
                if max_y > 90.0 {
                    max_y = 90.0;
                }
                if min_y < -90.0 {
                    min_y = -90.0;
                }
            } else {
                let lon_delta_deg =
                    Self::calc_box_by_dist_from_pt_delta_lon_deg(lat, lon, dist_deg);
                min_x = Self::norm_lon_deg(lon - lon_delta_deg);
                max_x = Self::norm_lon_deg(lon + lon_delta_deg);
            }
        }
        [min_x, max_x, min_y, max_y]
    }

    /// `calcBoxByDistFromPt_deltaLonDEG(lat, lon, distDEG)`.
    pub fn calc_box_by_dist_from_pt_delta_lon_deg(lat: f64, _lon: f64, dist_deg: f64) -> f64 {
        if dist_deg == 0.0 {
            return 0.0;
        }
        let lat_rad = Self::to_radians(lat);
        let dist_rad = Self::to_radians(dist_deg);
        let result_rad = asin(sin(dist_rad) / cos(lat_rad));
        if !result_rad.is_nan() {
            return Self::to_degrees(result_rad);
        }
        90.0
    }

    /// `calcBoxByDistFromPt_latHorizAxisDEG(lat, lon, distDEG)`.
    pub fn calc_box_by_dist_from_pt_lat_horiz_axis_deg(lat: f64, _lon: f64, dist_deg: f64) -> f64 {
        if dist_deg == 0.0 {
            return lat;
        } else if lat + dist_deg >= 90.0 {
            return 90.0;
        } else if lat - dist_deg <= -90.0 {
            return -90.0;
        }
        let lat_rad = Self::to_radians(lat);
        let dist_rad = Self::to_radians(dist_deg);
        let result_rad = asin(sin(lat_rad) / cos(dist_rad));
        if !result_rad.is_nan() {
            return Self::to_degrees(result_rad);
        }
        if lat > 0.0 {
            return 90.0;
        }
        if lat < 0.0 {
            return -90.0;
        }
        lat
    }

    /// `calcLonDegreesAtLat(lat, dist)`.
    pub fn calc_lon_degrees_at_lat(lat: f64, dist: f64) -> f64 {
        let distance_rad = Self::to_radians(dist);
        let start_lat = Self::to_radians(lat);
        let cos_ang_dist = cos(distance_rad);
        let cos_start_lat = cos(start_lat);
        let sin_ang_dist = sin(distance_rad);
        let sin_start_lat = sin(start_lat);
        let lon_delta = atan2(
            sin_ang_dist * cos_start_lat,
            cos_ang_dist * (1.0 - sin_start_lat * sin_start_lat),
        );
        Self::to_degrees(lon_delta)
    }

    /// `distHaversineRAD(lat1, lon1, lat2, lon2)`.
    pub fn dist_haversine_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        if lat1 == lat2 && lon1 == lon2 {
            return 0.0;
        }
        let hsin_x = sin((lon1 - lon2) * 0.5);
        let hsin_y = sin((lat1 - lat2) * 0.5);
        let mut h = hsin_y * hsin_y + (cos(lat1) * cos(lat2) * hsin_x * hsin_x);
        if h > 1.0 {
            h = 1.0;
        }
        2.0 * atan2(h.sqrt(), (1.0 - h).sqrt())
    }

    /// `distLawOfCosinesRAD(lat1, lon1, lat2, lon2)`.
    pub fn dist_law_of_cosines_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        if lat1 == lat2 && lon1 == lon2 {
            return 0.0;
        }
        let d_lon = lon2 - lon1;
        let cos_b = (sin(lat1) * sin(lat2)) + (cos(lat1) * cos(lat2) * cos(d_lon));
        if cos_b < -1.0 {
            PI
        } else if cos_b >= 1.0 {
            0.0
        } else {
            acos(cos_b)
        }
    }

    /// `distVincentyRAD(lat1, lon1, lat2, lon2)`.
    pub fn dist_vincenty_rad(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        if lat1 == lat2 && lon1 == lon2 {
            return 0.0;
        }
        let cos_lat1 = cos(lat1);
        let cos_lat2 = cos(lat2);
        let sin_lat1 = sin(lat1);
        let sin_lat2 = sin(lat2);
        let d_lon = lon2 - lon1;
        let cos_d_lon = cos(d_lon);
        let sin_d_lon = sin(d_lon);
        let a = cos_lat2 * sin_d_lon;
        let b = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_d_lon;
        let c = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_d_lon;
        atan2((a * a + b * b).sqrt(), c)
    }

    /// `dist2Degrees(dist, radius)`.
    pub fn dist2_degrees(dist: f64, radius: f64) -> f64 {
        Self::to_degrees(Self::dist2_radians(dist, radius))
    }

    /// `degrees2Dist(degrees, radius)`.
    pub fn degrees2_dist(degrees: f64, radius: f64) -> f64 {
        Self::radians2_dist(Self::to_radians(degrees), radius)
    }

    /// `dist2Radians(dist, radius)`.
    pub fn dist2_radians(dist: f64, radius: f64) -> f64 {
        dist / radius
    }

    /// `radians2Dist(radians, radius)`.
    pub fn radians2_dist(radians: f64, radius: f64) -> f64 {
        radians * radius
    }

    /// `toRadians(degrees)`: `degrees * DEGREES_TO_RADIANS` (not
    /// `Math.toRadians`, which rounds differently).
    pub fn to_radians(degrees: f64) -> f64 {
        degrees * Self::DEGREES_TO_RADIANS
    }

    /// `toDegrees(radians)`.
    pub fn to_degrees(radians: f64) -> f64 {
        radians * Self::RADIANS_TO_DEGREES
    }
}

/// `DistanceCalculator`. `Display` is Java's `toString()` (the class's
/// simple name).
pub trait DistanceCalculator: fmt::Display + Send + Sync + Any {
    /// `distance(from, to)`.
    fn distance(&self, from: &dyn Point, to: &dyn Point) -> Result<f64> {
        self.distance_xy(from, to.x(), to.y())
    }

    /// `distance(from, toX, toY)`.
    fn distance_xy(&self, from: &dyn Point, to_x: f64, to_y: f64) -> Result<f64>;

    /// `within(from, toX, toY, distance)`.
    fn within(&self, from: &dyn Point, to_x: f64, to_y: f64, distance: f64) -> Result<bool> {
        Ok(self.distance_xy(from, to_x, to_y)? <= distance)
    }

    /// `pointOnBearing(from, distDEG, bearingDEG, ctx, null)`.
    fn point_on_bearing(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        bearing_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Point>>;

    /// `calcBoxByDistFromPt(from, distDEG, ctx, null)`.
    fn calc_box_by_dist_from_pt(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Rectangle>>;

    /// `calcBoxByDistFromPt_yHorizAxisDEG(from, distDEG, ctx)`.
    fn calc_box_by_dist_from_pt_y_horiz_axis_deg(
        &self,
        from: &dyn Point,
        dist_deg: f64,
        ctx: &SpatialContext,
    ) -> Result<f64>;

    /// `area(rect)`.
    fn area_rect(&self, rect: &dyn Rectangle) -> Result<f64>;

    /// `area(circle)`.
    fn area_circle(&self, circle: &dyn Circle) -> Result<f64>;

    /// `equals(other)`.
    fn equals(&self, other: &dyn DistanceCalculator) -> bool;

    /// For downcasting.
    fn as_any(&self) -> &dyn Any;
}

/// `GeodesicSphereDistCalc`: great-circle distances on a sphere, in
/// degrees, by one of three formulas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeodesicSphereDistCalc {
    /// `GeodesicSphereDistCalc.Haversine` (the geo default).
    Haversine,
    /// `GeodesicSphereDistCalc.LawOfCosines`.
    LawOfCosines,
    /// `GeodesicSphereDistCalc.Vincenty`.
    Vincenty,
}

/// `GeodesicSphereDistCalc.radiusDEG`: one radian in degrees.
const RADIUS_DEG: f64 = 1.0 * DistanceUtils::RADIANS_TO_DEGREES;

impl GeodesicSphereDistCalc {
    /// `distanceLatLonRAD(lat1, lon1, lat2, lon2)`.
    pub fn distance_lat_lon_rad(self, lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
        match self {
            GeodesicSphereDistCalc::Haversine => {
                DistanceUtils::dist_haversine_rad(lat1, lon1, lat2, lon2)
            }
            GeodesicSphereDistCalc::LawOfCosines => {
                DistanceUtils::dist_law_of_cosines_rad(lat1, lon1, lat2, lon2)
            }
            GeodesicSphereDistCalc::Vincenty => {
                DistanceUtils::dist_vincenty_rad(lat1, lon1, lat2, lon2)
            }
        }
    }
}

impl fmt::Display for GeodesicSphereDistCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            GeodesicSphereDistCalc::Haversine => "Haversine",
            GeodesicSphereDistCalc::LawOfCosines => "LawOfCosines",
            GeodesicSphereDistCalc::Vincenty => "Vincenty",
        })
    }
}

impl DistanceCalculator for GeodesicSphereDistCalc {
    fn distance_xy(&self, from: &dyn Point, to_x: f64, to_y: f64) -> Result<f64> {
        let to_r = DistanceUtils::to_radians;
        Ok(DistanceUtils::to_degrees(self.distance_lat_lon_rad(
            to_r(from.y()),
            to_r(from.x()),
            to_r(to_y),
            to_r(to_x),
        )))
    }

    fn point_on_bearing(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        bearing_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Point>> {
        if dist_deg == 0.0 {
            return Ok(from.clone());
        }
        let to_r = DistanceUtils::to_radians;
        let (lon2, lat2) = DistanceUtils::point_on_bearing_rad(
            to_r(from.y()),
            to_r(from.x()),
            to_r(dist_deg),
            to_r(bearing_deg),
        );
        // Java makes the point in radians (verifying it), then resets it to
        // degrees (without verifying).
        ctx.point_xy(lon2, lat2)?;
        ctx.shape_factory().point_xy_unchecked(
            ctx,
            DistanceUtils::to_degrees(lon2),
            DistanceUtils::to_degrees(lat2),
        )
    }

    fn calc_box_by_dist_from_pt(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Rectangle>> {
        let [a, b, c, d] =
            DistanceUtils::calc_box_by_dist_from_pt_deg(from.y(), from.x(), dist_deg);
        ctx.rect(a, b, c, d)
    }

    fn calc_box_by_dist_from_pt_y_horiz_axis_deg(
        &self,
        from: &dyn Point,
        dist_deg: f64,
        _ctx: &SpatialContext,
    ) -> Result<f64> {
        Ok(DistanceUtils::calc_box_by_dist_from_pt_lat_horiz_axis_deg(
            from.y(),
            from.x(),
            dist_deg,
        ))
    }

    fn area_rect(&self, rect: &dyn Rectangle) -> Result<f64> {
        let lat1 = DistanceUtils::to_radians(rect.min_y());
        let lat2 = DistanceUtils::to_radians(rect.max_y());
        Ok(PI / 180.0 * RADIUS_DEG * RADIUS_DEG * (sin(lat1) - sin(lat2)).abs() * rect.width())
    }

    fn area_circle(&self, circle: &dyn Circle) -> Result<f64> {
        let lat = DistanceUtils::to_radians(90.0 - circle.radius());
        Ok(2.0 * PI * RADIUS_DEG * RADIUS_DEG * (1.0 - sin(lat)))
    }

    fn equals(&self, other: &dyn DistanceCalculator) -> bool {
        other.as_any().downcast_ref::<GeodesicSphereDistCalc>() == Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `CartesianDistCalc`: planar distance, optionally squared.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CartesianDistCalc {
    squared: bool,
}

impl CartesianDistCalc {
    /// `new CartesianDistCalc(squared)`.
    pub fn new(squared: bool) -> Self {
        CartesianDistCalc { squared }
    }

    fn distance_squared(from_x: f64, from_y: f64, to_x: f64, to_y: f64) -> f64 {
        let delta_x = from_x - to_x;
        let delta_y = from_y - to_y;
        delta_x * delta_x + delta_y * delta_y
    }

    /// `distanceToLineSegment(point, vX, vY, wX, wY)`.
    pub fn distance_to_line_segment(
        &self,
        point: &dyn Point,
        v_x: f64,
        v_y: f64,
        w_x: f64,
        w_y: f64,
    ) -> f64 {
        let d = Self::distance_squared(v_x, v_y, w_x, w_y);
        let (to_x, to_y);
        if d <= 0.0 {
            to_x = v_x;
            to_y = v_y;
        } else {
            let t = ((point.x() - v_x) * (w_x - v_x) + (point.y() - v_y) * (w_y - v_y)) / d;
            if t < 0.0 {
                to_x = v_x;
                to_y = v_y;
            } else if t > 1.0 {
                to_x = w_x;
                to_y = w_y;
            } else {
                to_x = v_x + t * (w_x - v_x);
                to_y = v_y + t * (w_y - v_y);
            }
        }
        self.distance_xy_plain(point.x(), point.y(), to_x, to_y)
    }

    fn distance_xy_plain(&self, from_x: f64, from_y: f64, to_x: f64, to_y: f64) -> f64 {
        let x_squared_plus_y_squared = Self::distance_squared(from_x, from_y, to_x, to_y);
        if self.squared {
            return x_squared_plus_y_squared;
        }
        x_squared_plus_y_squared.sqrt()
    }
}

impl fmt::Display for CartesianDistCalc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CartesianDistCalc")
    }
}

impl DistanceCalculator for CartesianDistCalc {
    fn distance_xy(&self, from: &dyn Point, to_x: f64, to_y: f64) -> Result<f64> {
        Ok(self.distance_xy_plain(from.x(), from.y(), to_x, to_y))
    }

    fn within(&self, from: &dyn Point, to_x: f64, to_y: f64, distance: f64) -> Result<bool> {
        let delta_x = from.x() - to_x;
        let delta_y = from.y() - to_y;
        Ok(delta_x * delta_x + delta_y * delta_y <= distance * distance)
    }

    fn point_on_bearing(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        bearing_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Point>> {
        if dist_deg == 0.0 {
            return Ok(from.clone());
        }
        let bearing_rad = DistanceUtils::to_radians(bearing_deg);
        let x = from.x() + sin(bearing_rad) * dist_deg;
        let y = from.y() + cos(bearing_rad) * dist_deg;
        ctx.point_xy(x, y)
    }

    fn calc_box_by_dist_from_pt(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Rectangle>> {
        let min_x = from.x() - dist_deg;
        let max_x = from.x() + dist_deg;
        let min_y = from.y() - dist_deg;
        let max_y = from.y() + dist_deg;
        ctx.rect(min_x, max_x, min_y, max_y)
    }

    fn calc_box_by_dist_from_pt_y_horiz_axis_deg(
        &self,
        from: &dyn Point,
        _dist_deg: f64,
        _ctx: &SpatialContext,
    ) -> Result<f64> {
        Ok(from.y())
    }

    fn area_rect(&self, rect: &dyn Rectangle) -> Result<f64> {
        rect.area(None)
    }

    fn area_circle(&self, circle: &dyn Circle) -> Result<f64> {
        circle.area(None)
    }

    fn equals(&self, other: &dyn DistanceCalculator) -> bool {
        other.as_any().downcast_ref::<CartesianDistCalc>() == Some(self)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pole_reflection_flips_longitude() {
        let (lon, lat) = DistanceUtils::normalize_lon_lat_rad(1.0, 2.0);
        assert_eq!(lat, PI - 2.0);
        assert_eq!(lon, 1.0 - PI);
        let (lon, lat) = DistanceUtils::normalize_lon_lat_rad(-1.0, -2.0);
        assert_eq!(lat, -PI + 2.0);
        assert_eq!(lon, -1.0 + PI);
        let (lon, _) = DistanceUtils::normalize_lon_lat_rad(-1.0, 2.0);
        assert_eq!(lon, -1.0 + PI);
        let (lon, _) = DistanceUtils::normalize_lon_lat_rad(1.0, -2.0);
        assert_eq!(lon, 1.0 - PI);
    }

    #[test]
    fn horizontal_axis_and_box_edges() {
        let f = DistanceUtils::calc_box_by_dist_from_pt_lat_horiz_axis_deg;
        assert_eq!(f(10.0, 0.0, f64::NAN), 90.0);
        assert_eq!(f(-10.0, 0.0, f64::NAN), -90.0);
        assert!(f(f64::NAN, 0.0, 1.0).is_nan());
        // touching (not passing) a pole spans 180 degrees of longitude
        assert_eq!(
            DistanceUtils::calc_box_by_dist_from_pt_deg(45.0, 10.0, 45.0),
            [-80.0, 100.0, 0.0, 90.0]
        );
        assert_eq!(
            DistanceUtils::calc_box_by_dist_from_pt_delta_lon_deg(89.0, 0.0, 10.0),
            90.0
        );
    }
}
