//! Port of `org.apache.lucene.geo.GeoUtils`.

use super::rectangle::Rectangle;
use super::{java_double_string, GeoError, Relation};
use crate::sloppy_math;

/// Port of `org.apache.lucene.geo.GeoUtils`: coordinate validation and the
/// shared geometric predicates.
#[derive(Debug, Clone, Copy)]
pub struct GeoUtils;

/// `GeoUtils.WindingOrder`: the orientation of three points.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindingOrder {
    /// Clockwise (sign -1).
    CW,
    /// Colinear (sign 0).
    Colinear,
    /// Counter-clockwise (sign 1).
    CCW,
}

impl WindingOrder {
    /// `sign()`.
    // SENTINEL: none -- `-1` is `CW`'s sign, a value in the domain, not an
    // out-of-band marker.
    pub fn sign(self) -> i32 {
        match self {
            WindingOrder::CW => -1,
            WindingOrder::Colinear => 0,
            WindingOrder::CCW => 1,
        }
    }

    /// `fromSign(int)`.
    pub fn from_sign(sign: i32) -> Result<WindingOrder, GeoError> {
        match sign {
            -1 => Ok(WindingOrder::CW),
            0 => Ok(WindingOrder::Colinear),
            1 => Ok(WindingOrder::CCW),
            _ => Err(GeoError::illegal(format!(
                "Invalid WindingOrder sign: {sign}"
            ))),
        }
    }
}

impl GeoUtils {
    /// `MIN_LON_INCL`.
    pub const MIN_LON_INCL: f64 = -180.0;
    /// `MAX_LON_INCL`.
    pub const MAX_LON_INCL: f64 = 180.0;
    /// `MIN_LAT_INCL`.
    pub const MIN_LAT_INCL: f64 = -90.0;
    /// `MAX_LAT_INCL`.
    pub const MAX_LAT_INCL: f64 = 90.0;
    /// `EARTH_MEAN_RADIUS_METERS`.
    pub const EARTH_MEAN_RADIUS_METERS: f64 = 6_371_008.771_4;

    /// `MIN_LON_RADIANS`.
    pub fn min_lon_radians() -> f64 {
        Self::MIN_LON_INCL.to_radians()
    }
    /// `MAX_LON_RADIANS`.
    pub fn max_lon_radians() -> f64 {
        Self::MAX_LON_INCL.to_radians()
    }
    /// `MIN_LAT_RADIANS`.
    pub fn min_lat_radians() -> f64 {
        Self::MIN_LAT_INCL.to_radians()
    }
    /// `MAX_LAT_RADIANS`.
    pub fn max_lat_radians() -> f64 {
        Self::MAX_LAT_INCL.to_radians()
    }

    /// `checkLatitude`: within `[-90, 90]` and not NaN.
    pub fn check_latitude(latitude: f64) -> Result<(), GeoError> {
        if !(Self::MIN_LAT_INCL..=Self::MAX_LAT_INCL).contains(&latitude) {
            return Err(GeoError::illegal(format!(
                "invalid latitude {}; must be between {} and {}",
                java_double_string(latitude),
                java_double_string(Self::MIN_LAT_INCL),
                java_double_string(Self::MAX_LAT_INCL)
            )));
        }
        Ok(())
    }

    /// `checkLongitude`: within `[-180, 180]` and not NaN.
    pub fn check_longitude(longitude: f64) -> Result<(), GeoError> {
        if !(Self::MIN_LON_INCL..=Self::MAX_LON_INCL).contains(&longitude) {
            return Err(GeoError::illegal(format!(
                "invalid longitude {}; must be between {} and {}",
                java_double_string(longitude),
                java_double_string(Self::MIN_LON_INCL),
                java_double_string(Self::MAX_LON_INCL)
            )));
        }
        Ok(())
    }

    /// `distanceQuerySortKey`: binary search for the exact sort key a radius
    /// corresponds to; any sort key `<=` it is a match.
    pub fn distance_query_sort_key(radius: f64) -> f64 {
        let max = sloppy_math::haversin_meters_from_sort_key(f64::MAX);
        if radius >= max {
            return max;
        }
        // a search through non-negative long space only
        let mut lo: i64 = 0;
        let mut hi: i64 = f64::MAX.to_bits() as i64;
        while lo <= hi {
            let mid = ((lo as u64).wrapping_add(hi as u64) >> 1) as i64;
            let sort_key = f64::from_bits(mid as u64);
            let mid_radius = sloppy_math::haversin_meters_from_sort_key(sort_key);
            if mid_radius == radius {
                return sort_key;
            } else if mid_radius > radius {
                hi = mid.wrapping_sub(1);
            } else {
                lo = mid.wrapping_add(1);
            }
        }
        f64::from_bits(lo as u64)
    }

    /// `relate`: the relation between a box (not crossing the dateline) and
    /// a distance query.
    #[allow(clippy::too_many_arguments)]
    pub fn relate(
        min_lat: f64,
        max_lat: f64,
        min_lon: f64,
        max_lon: f64,
        lat: f64,
        lon: f64,
        distance_sort_key: f64,
        axis_lat: f64,
    ) -> Result<Relation, GeoError> {
        if min_lon > max_lon {
            return Err(GeoError::illegal("Box crosses the dateline"));
        }
        let key = sloppy_math::haversin_sort_key;
        if (lon < min_lon || lon > max_lon)
            && (axis_lat + Rectangle::AXISLAT_ERROR < min_lat
                || axis_lat - Rectangle::AXISLAT_ERROR > max_lat)
        {
            // circle not fully inside / crossing axis
            if key(lat, lon, min_lat, min_lon) > distance_sort_key
                && key(lat, lon, min_lat, max_lon) > distance_sort_key
                && key(lat, lon, max_lat, min_lon) > distance_sort_key
                && key(lat, lon, max_lat, max_lon) > distance_sort_key
            {
                return Ok(Relation::CellOutsideQuery);
            }
        }
        if Self::within_90_lon_degrees(lon, min_lon, max_lon)
            && key(lat, lon, min_lat, min_lon) <= distance_sort_key
            && key(lat, lon, min_lat, max_lon) <= distance_sort_key
            && key(lat, lon, max_lat, min_lon) <= distance_sort_key
            && key(lat, lon, max_lat, max_lon) <= distance_sort_key
        {
            return Ok(Relation::CellInsideQuery);
        }
        Ok(Relation::CellCrossesQuery)
    }

    /// `within90LonDegrees`.
    pub(crate) fn within_90_lon_degrees(mut lon: f64, min_lon: f64, max_lon: f64) -> bool {
        if max_lon <= lon - 180.0 {
            lon -= 360.0;
        } else if min_lon >= lon + 180.0 {
            lon += 360.0;
        }
        max_lon - lon < 90.0 && lon - min_lon < 90.0
    }

    /// `orient`: positive if a, b, c are counter-clockwise, negative if
    /// clockwise, zero if collinear.
    // SENTINEL: none -- `-1` means "clockwise", one of the three results,
    // not an out-of-band marker.
    #[inline]
    pub fn orient(ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> i32 {
        let v1 = (bx - ax) * (cy - ay);
        let v2 = (cx - ax) * (by - ay);
        if v1 > v2 {
            1
        } else if v1 < v2 {
            -1
        } else {
            0
        }
    }

    /// `lineCrossesLine`: the segments cross, end points excluded.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn line_crosses_line(
        a1x: f64,
        a1y: f64,
        b1x: f64,
        b1y: f64,
        a2x: f64,
        a2y: f64,
        b2x: f64,
        b2y: f64,
    ) -> bool {
        Self::orient(a2x, a2y, b2x, b2y, a1x, a1y) * Self::orient(a2x, a2y, b2x, b2y, b1x, b1y) < 0
            && Self::orient(a1x, a1y, b1x, b1y, a2x, a2y)
                * Self::orient(a1x, a1y, b1x, b1y, b2x, b2y)
                < 0
    }

    /// `lineOverlapLine`: the segments are collinear.
    #[allow(clippy::too_many_arguments)]
    pub fn line_overlap_line(
        a1x: f64,
        a1y: f64,
        b1x: f64,
        b1y: f64,
        a2x: f64,
        a2y: f64,
        b2x: f64,
        b2y: f64,
    ) -> bool {
        Self::orient(a2x, a2y, b2x, b2y, a1x, a1y) == 0
            && Self::orient(a2x, a2y, b2x, b2y, b1x, b1y) == 0
            && Self::orient(a1x, a1y, b1x, b1y, a2x, a2y) == 0
            && Self::orient(a1x, a1y, b1x, b1y, b2x, b2y) == 0
    }

    /// `lineCrossesLineWithBoundary`: the segments cross or touch.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn line_crosses_line_with_boundary(
        a1x: f64,
        a1y: f64,
        b1x: f64,
        b1y: f64,
        a2x: f64,
        a2y: f64,
        b2x: f64,
        b2y: f64,
    ) -> bool {
        Self::orient(a2x, a2y, b2x, b2y, a1x, a1y) * Self::orient(a2x, a2y, b2x, b2y, b1x, b1y) <= 0
            && Self::orient(a1x, a1y, b1x, b1y, a2x, a2y)
                * Self::orient(a1x, a1y, b1x, b1y, b2x, b2y)
                <= 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checks() {
        assert!(GeoUtils::check_latitude(90.0).is_ok());
        assert_eq!(
            GeoUtils::check_latitude(90.5).unwrap_err().to_string(),
            "invalid latitude 90.5; must be between -90.0 and 90.0"
        );
        assert!(GeoUtils::check_latitude(f64::NAN).is_err());
        assert!(GeoUtils::check_longitude(-180.0).is_ok());
        assert_eq!(
            GeoUtils::check_longitude(-181.0).unwrap_err().to_string(),
            "invalid longitude -181.0; must be between -180.0 and 180.0"
        );
        assert!(GeoUtils::check_longitude(f64::NAN).is_err());
    }

    #[test]
    fn winding_order() {
        for w in [WindingOrder::CW, WindingOrder::Colinear, WindingOrder::CCW] {
            assert_eq!(WindingOrder::from_sign(w.sign()).unwrap(), w);
        }
        assert_eq!(
            WindingOrder::from_sign(2).unwrap_err().to_string(),
            "Invalid WindingOrder sign: 2"
        );
    }

    #[test]
    fn relate_rejects_dateline_boxes() {
        assert!(GeoUtils::relate(0.0, 1.0, 10.0, -10.0, 0.0, 0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn sort_key_search() {
        let max = GeoUtils::distance_query_sort_key(f64::INFINITY);
        assert_eq!(max, sloppy_math::haversin_meters_from_sort_key(f64::MAX));
        for r in [0.0, 1.0, 1000.0, 123_456.789] {
            let k = GeoUtils::distance_query_sort_key(r);
            assert!(sloppy_math::haversin_meters_from_sort_key(k) >= r);
        }
    }

    #[test]
    fn segments() {
        assert!(GeoUtils::line_crosses_line(
            0.0, 0.0, 2.0, 2.0, 0.0, 2.0, 2.0, 0.0
        ));
        assert!(!GeoUtils::line_crosses_line(
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0
        ));
        assert!(GeoUtils::line_crosses_line_with_boundary(
            0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0
        ));
        assert!(GeoUtils::line_overlap_line(
            0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 3.0
        ));
        assert!(!GeoUtils::line_overlap_line(
            0.0, 0.0, 2.0, 2.0, 1.0, 1.0, 3.0, 4.0
        ));
        assert!(!GeoUtils::within_90_lon_degrees(170.0, -170.0, -100.0));
        assert!(GeoUtils::within_90_lon_degrees(-170.0, 170.0, 179.0));
        assert!(GeoUtils::within_90_lon_degrees(170.0, -179.0, -170.0));
    }
}
