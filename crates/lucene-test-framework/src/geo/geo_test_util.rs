//! `GeoTestUtil`'s point, box and circle generators: values that favour
//! the edges, zero, a 360-way grid and one-ulp neighbours.
//!
//! Java draws from the test's `random()`; here each takes the stream.
//! The polygon and line generators (`nextPolygon`, `createRegularPolygon`,
//! `nextPointNear`, `nextBoxNear`) and `toSVG` are not ported (see the
//! ledger).

use lucene_util::geo::GeoUtils;
use lucene_util::java_random::JavaRandom;
use lucene_util::numeric_utils::{double_to_sortable_long, sortable_long_to_double};

use crate::util::test_util;

/// `Math.nextAfter(float start, double direction)`, widened back to a
/// double: the overload Java resolves `nextAfter(int, double)` to, since
/// `float` is the more specific widening of an `int`.
#[must_use]
pub fn java_next_after_f32(start: f32, direction: f64) -> f64 {
    let s = f64::from(start);
    if s < direction {
        f64::from(start.next_up())
    } else if s > direction {
        f64::from(start.next_down())
    } else {
        direction
    }
}

/// `nextDoubleInternal(low, high)`: a base value (random bits, either
/// edge, zero, one of 360 blocks, or uniform), then Java's one-ulp nudge --
/// which, as Java's does, nudges the draw index (`0` or `1`, as a float),
/// not the base value.
///
/// # Panics
/// When `high < low`.
pub fn next_double_internal(random: &mut JavaRandom, low: f64, high: f64) -> f64 {
    assert!(high >= low, "low={low} high={high}");
    #[allow(clippy::float_cmp)]
    if low == high {
        return low;
    }
    let base_value = match random.next_int_bounded(17) {
        0 => sortable_long_to_double(test_util::next_long(
            random,
            double_to_sortable_long(low),
            double_to_sortable_long(high),
        )),
        1 => low,
        2 => high,
        3 if low <= 0.0 && high >= 0.0 => 0.0,
        4 => {
            let delta = (high - low) / 360.0;
            let block = random.next_int_bounded(360);
            low + delta * f64::from(block)
        }
        _ => low + (high - low) * random.next_double(),
    };
    match random.next_int_bounded(17) {
        0 => java_next_after_f32(0.0, high),
        1 => java_next_after_f32(1.0, low),
        _ => base_value,
    }
}

/// `nextLatitude()`.
pub fn next_latitude(random: &mut JavaRandom) -> f64 {
    next_double_internal(random, GeoUtils::MIN_LAT_INCL, GeoUtils::MAX_LAT_INCL)
}

/// `nextLongitude()`.
pub fn next_longitude(random: &mut JavaRandom) -> f64 {
    next_double_internal(random, GeoUtils::MIN_LON_INCL, GeoUtils::MAX_LON_INCL)
}

/// `Rectangle(minLat, maxLat, minLon, maxLon)`: a box, crossing the
/// dateline when `max_lon < min_lon`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    /// `minLat`.
    pub min_lat: f64,
    /// `maxLat`.
    pub max_lat: f64,
    /// `minLon`.
    pub min_lon: f64,
    /// `maxLon`.
    pub max_lon: f64,
}

/// `nextBoxInternal(canCrossDateLine)`: never a line.
fn next_box_internal(random: &mut JavaRandom, can_cross_date_line: bool) -> Rect {
    let mut lat0 = next_latitude(random);
    let mut lat1 = next_latitude(random);
    #[allow(clippy::float_cmp)]
    while lat0 == lat1 {
        lat1 = next_latitude(random);
    }
    let mut lon0 = next_longitude(random);
    let mut lon1 = next_longitude(random);
    #[allow(clippy::float_cmp)]
    while lon0 == lon1 {
        lon1 = next_longitude(random);
    }
    if lat1 < lat0 {
        std::mem::swap(&mut lat0, &mut lat1);
    }
    if !can_cross_date_line && lon1 < lon0 {
        std::mem::swap(&mut lon0, &mut lon1);
    }
    Rect {
        min_lat: lat0,
        max_lat: lat1,
        min_lon: lon0,
        max_lon: lon1,
    }
}

/// `nextBox()`: may cross the dateline.
pub fn next_box(random: &mut JavaRandom) -> Rect {
    next_box_internal(random, true)
}

/// `nextBoxNotCrossingDateline()`.
pub fn next_box_not_crossing_dateline(random: &mut JavaRandom) -> Rect {
    next_box_internal(random, false)
}

/// `nextCircle()`'s radius: up to a quarter of the earth's circumference
/// (so at most half its surface), plus a metre.
pub fn next_radius_meters(random: &mut JavaRandom) -> f64 {
    random.next_double() * GeoUtils::EARTH_MEAN_RADIUS_METERS * std::f64::consts::PI / 2.0 + 1.0
}

/// `nextCircle()`: `(lat, lon, radiusMeters)`.
pub fn next_circle(random: &mut JavaRandom) -> (f64, f64, f64) {
    let lat = next_latitude(random);
    let lon = next_longitude(random);
    (lat, lon, next_radius_meters(random))
}
