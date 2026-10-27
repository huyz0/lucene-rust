//! `GeoTestUtil`'s point, box and circle generators and slow geographic
//! polygon containment oracle. Generated values favour the edges, zero,
//! a 360-way grid and one-ulp neighbours.
//!
//! Java draws from the test's `random()`; here each takes the stream.
//! The containment oracle borrows immutable polygon coordinates instead
//! of Java's cloned getter arrays, and returns a typed error for holes.
//! Its boundary tests and floating-point operation order remain Java's.
//!
//! The polygon, line and nearby-shape generators keep Java's draws call
//! for call; `Math.sin`/`cos` are `StrictMath`'s (the fixture generator
//! runs with HotSpot's trig intrinsics off, as the geo3d ones do). An
//! exception Java lets escape (a `Polygon` constructor's on a degenerate
//! ring, `checkLatitude` on an invalid anchor) is an `Err` or a panic with
//! its message. `toSVG` (a debugging aid) and `readShape` (classpath
//! resources of the Java tests) are `not-needed`.

use lucene_util::geo::{Circle, GeoError, GeoUtils, Line, Point, Polygon};
use lucene_util::java_random::JavaRandom;
use lucene_util::numeric_utils::{double_to_sortable_long, sortable_long_to_double};
use lucene_util::sloppy_math::haversin_meters;
use lucene_util::strict_math;

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

/// `Math.min(a, b)`: `NaN` wins, and `-0.0` is below `0.0`.
fn java_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_negative() {
            a
        } else {
            b
        }
    } else if a <= b {
        a
    } else {
        b
    }
}

/// `Math.max(a, b)`.
fn java_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a == 0.0 && b == 0.0 {
        if a.is_sign_positive() {
            a
        } else {
            b
        }
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `GeoUtils.checkLatitude`, whose exception Java's generators let escape.
fn check_latitude(lat: f64) {
    if let Err(e) = GeoUtils::check_latitude(lat) {
        panic!("{e}");
    }
}

/// `GeoUtils.checkLongitude`.
fn check_longitude(lon: f64) {
    if let Err(e) = GeoUtils::check_longitude(lon) {
        panic!("{e}");
    }
}

/// `nextLatitudeNear(otherLatitude, delta)`.
fn next_latitude_near(random: &mut JavaRandom, other: f64, delta: f64) -> f64 {
    let delta = delta.abs();
    check_latitude(other);
    let surprise_me = random.next_int_bounded(97);
    if surprise_me == 0 {
        next_latitude(random)
    } else if surprise_me < 49 {
        next_double_internal(random, other, java_min(90.0, other + delta))
    } else {
        next_double_internal(random, java_max(-90.0, other - delta), other)
    }
}

/// `nextLongitudeNear(otherLongitude, delta)`.
fn next_longitude_near(random: &mut JavaRandom, other: f64, delta: f64) -> f64 {
    let delta = delta.abs();
    check_longitude(other);
    let surprise_me = random.next_int_bounded(97);
    if surprise_me == 0 {
        next_longitude(random)
    } else if surprise_me < 49 {
        next_double_internal(random, other, java_min(180.0, other + delta))
    } else {
        next_double_internal(random, java_max(-180.0, other - delta), other)
    }
}

/// `nextLatitudeBetween(min, max)`: the range widened by 1%, or now and
/// then anywhere.
fn next_latitude_between(random: &mut JavaRandom, min: f64, max: f64) -> f64 {
    check_latitude(min);
    check_latitude(max);
    if random.next_int_bounded(47) == 0 {
        next_latitude(random)
    } else {
        let difference = (max - min) / 100.0;
        let lower = java_max(-90.0, min - difference);
        let upper = java_min(90.0, max + difference);
        next_double_internal(random, lower, upper)
    }
}

/// `nextLongitudeBetween(min, max)`.
fn next_longitude_between(random: &mut JavaRandom, min: f64, max: f64) -> f64 {
    check_longitude(min);
    check_longitude(max);
    if random.next_int_bounded(47) == 0 {
        next_longitude(random)
    } else {
        let difference = (max - min) / 100.0;
        let lower = java_max(-180.0, min - difference);
        let upper = java_min(180.0, max + difference);
        next_double_internal(random, lower, upper)
    }
}

/// `nextPointAroundLine(lat1, lon1, lat2, lon2)`: `[lat, lon]`.
fn next_point_around_line(
    random: &mut JavaRandom,
    lat1: f64,
    lon1: f64,
    lat2: f64,
    lon2: f64,
) -> [f64; 2] {
    let (x1, x2, y1, y2) = (lon1, lon2, lat1, lat2);
    let min_x = java_min(x1, x2);
    let max_x = java_max(x1, x2);
    let min_y = java_min(y1, y2);
    let max_y = java_max(y1, y2);
    #[allow(clippy::float_cmp)]
    if min_x == max_x {
        let lat = next_latitude_between(random, min_y, max_y);
        [
            lat,
            next_longitude_near(random, min_x, 0.01 * (max_y - min_y)),
        ]
    } else if min_y == max_y {
        let lat = next_latitude_near(random, min_y, 0.01 * (max_x - min_x));
        [lat, next_longitude_between(random, min_x, max_x)]
    } else {
        let x = next_longitude_between(random, min_x, max_x);
        let mut y = (y1 - y2) / (x1 - x2) * (x - x1) + y1;
        if !y.is_finite() {
            // Underflow when the x values are wonderfully close.
            y = 90f64.copysign(x1);
        }
        let delta = (max_y - min_y) * 0.01;
        y = java_min(90.0, y);
        y = java_max(-90.0, y);
        [next_latitude_near(random, y, delta), x]
    }
}

/// `nextPointNear(Rectangle)`: `[lat, lon]`; a box across the dateline
/// first picks one of its halves.
pub fn next_point_near_box(random: &mut JavaRandom, rect: &Rect) -> [f64; 2] {
    if rect.max_lon < rect.min_lon {
        let half = if random.next_boolean() {
            Rect {
                min_lon: -180.0,
                ..*rect
            }
        } else {
            Rect {
                max_lon: 180.0,
                ..*rect
            }
        };
        next_point_near_box(random, &half)
    } else {
        next_point_near(random, &box_polygon(rect))
    }
}

/// `nextPointNear(Polygon)`: `[lat, lon]` near a hole, a vertex, the
/// bounding box's edges or a diagonal between vertices.
pub fn next_point_near(random: &mut JavaRandom, polygon: &Polygon) -> [f64; 2] {
    let poly_lats = polygon.poly_lats();
    let poly_lons = polygon.poly_lons();
    let holes = polygon.holes();
    if !holes.is_empty() && random.next_int_bounded(3) == 0 {
        let hole = &holes[index(random, holes.len())];
        return next_point_near(random, hole);
    }
    let surprise_me = random.next_int_bounded(97);
    if surprise_me == 0 {
        let lat = next_latitude(random);
        [lat, next_longitude(random)]
    } else if surprise_me < 5 {
        let lat = next_latitude_between(random, polygon.min_lat, polygon.max_lat);
        [
            lat,
            next_longitude_between(random, polygon.min_lon, polygon.max_lon),
        ]
    } else if surprise_me < 20 {
        let vertex = index(random, poly_lats.len() - 1);
        let lat = next_latitude_near(
            random,
            poly_lats[vertex],
            poly_lats[vertex + 1] - poly_lats[vertex],
        );
        let lon = next_longitude_near(
            random,
            poly_lons[vertex],
            poly_lons[vertex + 1] - poly_lons[vertex],
        );
        [lat, lon]
    } else if surprise_me < 30 {
        let container = box_polygon(&Rect {
            min_lat: polygon.min_lat,
            max_lat: polygon.max_lat,
            min_lon: polygon.min_lon,
            max_lon: polygon.max_lon,
        });
        let (lats, lons) = (container.poly_lats(), container.poly_lons());
        let start = index(random, lats.len() - 1);
        next_point_around_line(
            random,
            lats[start],
            lons[start],
            lats[start + 1],
            lons[start + 1],
        )
    } else {
        let start = index(random, poly_lats.len() - 1);
        // Favour edges heavily.
        let end = if random.next_boolean() {
            start + 1
        } else {
            index(random, poly_lats.len() - 1)
        };
        next_point_around_line(
            random,
            poly_lats[start],
            poly_lons[start],
            poly_lats[end],
            poly_lons[end],
        )
    }
}

/// `nextBoxNear(Polygon)`: a box from one interesting point and another
/// within an edge's length of it (or, rarely, two interesting points).
pub fn next_box_near(random: &mut JavaRandom, polygon: &Polygon) -> Rect {
    let holes = polygon.holes();
    if !holes.is_empty() && random.next_int_bounded(3) == 0 {
        let hole = &holes[index(random, holes.len())];
        return next_box_near(random, hole);
    }
    let surprise_me = random.next_int_bounded(97);
    let (point1, point2) = if surprise_me == 0 {
        let p1 = next_point_near(random, polygon);
        (p1, next_point_near(random, polygon))
    } else {
        let p1 = next_point_near(random, polygon);
        let (lats, lons) = (polygon.poly_lats(), polygon.poly_lons());
        let vertex = index(random, lats.len() - 1);
        let delta_x = lons[vertex + 1] - lons[vertex];
        let delta_y = lats[vertex + 1] - lats[vertex];
        let edge_length = (delta_x * delta_x + delta_y * delta_y).sqrt();
        let lat = next_latitude_near(random, p1[0], edge_length);
        (p1, [lat, next_longitude_near(random, p1[1], edge_length)])
    };
    Rect {
        min_lat: java_min(point1[0], point2[0]),
        max_lat: java_max(point1[0], point2[0]),
        min_lon: java_min(point1[1], point2[1]),
        max_lon: java_max(point1[1], point2[1]),
    }
}

/// `random.nextInt(bound)` as an index.
fn index(random: &mut JavaRandom, bound: usize) -> usize {
    let b = i32::try_from(bound).expect("a Java array length");
    usize::try_from(random.next_int_bounded(b)).expect("non-negative")
}

/// `createRegularPolygon(centerLat, centerLon, radiusMeters, gons)`: each
/// vertex walked out along its spoke until it is within 10 cm of
/// `radius_meters` from the centre.
///
/// # Errors
/// A spoke leaves the valid latitudes or longitudes (Java's
/// `IllegalArgumentException` from `checkLatitude`/`checkLongitude`).
pub fn create_regular_polygon(
    center_lat: f64,
    center_lon: f64,
    radius_meters: f64,
    gons: usize,
) -> Result<Polygon, GeoError> {
    let mut lats = vec![0.0; gons + 1];
    let mut lons = vec![0.0; gons + 1];
    for i in 0..gons {
        let angle = 360.0 - i as f64 * (360.0 / gons as f64);
        let x = strict_math::cos(angle.to_radians());
        let y = strict_math::sin(angle.to_radians());
        let mut factor = 2.0;
        let mut step = 1.0;
        let mut last = 0;
        loop {
            let lat = center_lat + y * factor;
            GeoUtils::check_latitude(lat)?;
            let lon = center_lon + x * factor;
            GeoUtils::check_longitude(lon)?;
            let distance_meters = haversin_meters(center_lat, center_lon, lat, lon);
            if (distance_meters - radius_meters).abs() < 0.1 {
                lats[i] = lat;
                lons[i] = lon;
                break;
            }
            if distance_meters > radius_meters {
                factor -= step;
                if last == 1 {
                    step /= 2.0;
                }
                last = -1;
            } else if distance_meters < radius_meters {
                factor += step;
                if last == -1 {
                    step /= 2.0;
                }
                last = 1;
            }
        }
    }
    lats[gons] = lats[0];
    lons[gons] = lons[0];
    Polygon::new(&lats, &lons, vec![])
}

/// `nextPoint()`.
pub fn next_point(random: &mut JavaRandom) -> Point {
    let lat = next_latitude(random);
    let lon = next_longitude(random);
    Point::new(lat, lon).expect("a valid point")
}

/// `nextLine()`: a random polygon's ring without its closing point.
///
/// # Errors
/// As [`next_polygon`].
pub fn next_line(random: &mut JavaRandom) -> Result<Line, GeoError> {
    let p = next_polygon(random)?;
    let n = p.poly_lats().len() - 1;
    Line::new(&p.poly_lats()[..n], &p.poly_lons()[..n])
}

/// `nextCircle()` as Lucene's `Circle`.
pub fn next_circle_shape(random: &mut JavaRandom) -> Circle {
    let (lat, lon, radius) = next_circle(random);
    Circle::new(lat, lon, radius).expect("a valid circle")
}

/// `nextPolygon()`: half the time an irregular ring, a tenth of the rest
/// a regular polygon (retried until it stays off the poles and the
/// dateline), else a random box or the triangle under its diagonal.
///
/// # Errors
/// An irregular ring of fewer than three vertices (Java's `Polygon`
/// constructor throws).
pub fn next_polygon(random: &mut JavaRandom) -> Result<Polygon, GeoError> {
    if random.next_boolean() {
        return surprise_me_polygon(random);
    } else if random.next_int_bounded(10) == 1 {
        loop {
            let gons = test_util::next_int(random, 4, 500);
            let radius_meters = next_radius_meters(random);
            let lat = next_latitude(random);
            let lon = next_longitude(random);
            let gons = usize::try_from(gons).expect("4..=500");
            if let Ok(p) = create_regular_polygon(lat, lon, radius_meters, gons) {
                return Ok(p);
            }
        }
    }
    let rect = next_box_internal(random, false);
    if random.next_boolean() {
        Ok(box_polygon(&rect))
    } else {
        Ok(triangle_polygon(&rect))
    }
}

/// `boxPolygon(box)`.
fn box_polygon(r: &Rect) -> Polygon {
    debug_assert!(r.max_lon >= r.min_lon, "the box crosses the dateline");
    Polygon::new(
        &[r.min_lat, r.max_lat, r.max_lat, r.min_lat, r.min_lat],
        &[r.min_lon, r.min_lon, r.max_lon, r.max_lon, r.min_lon],
        vec![],
    )
    .expect("a box is a polygon")
}

/// `trianglePolygon(box)`.
fn triangle_polygon(r: &Rect) -> Polygon {
    Polygon::new(
        &[r.min_lat, r.max_lat, r.max_lat, r.min_lat],
        &[r.min_lon, r.min_lon, r.max_lon, r.min_lon],
        vec![],
    )
    .expect("a triangle is a polygon")
}

/// `surpriseMePolygon()`: vertices at random angles and radii around a
/// random centre, restarted whenever one would cross a pole or the
/// dateline.
fn surprise_me_polygon(random: &mut JavaRandom) -> Result<Polygon, GeoError> {
    'new_poly: loop {
        let center_lat = next_latitude(random);
        let center_lon = next_longitude(random);
        let radius = 0.1 + 20.0 * random.next_double();
        let radius_delta = random.next_double();
        let mut lats = Vec::new();
        let mut lons = Vec::new();
        let mut angle = 0.0;
        loop {
            angle += random.next_double() * 40.0;
            if angle > 360.0 {
                break;
            }
            let len = radius * (1.0 - radius_delta + radius_delta * random.next_double());
            let lat = center_lat + len * strict_math::cos(f64::to_radians(angle));
            let lon = center_lon + len * strict_math::sin(f64::to_radians(angle));
            // Java's comparisons, which let a NaN through.
            #[allow(clippy::manual_range_contains)]
            if lon <= GeoUtils::MIN_LON_INCL
                || lon >= GeoUtils::MAX_LON_INCL
                || lat > 90.0
                || lat < -90.0
            {
                continue 'new_poly;
            }
            lats.push(lat);
            lons.push(lon);
        }
        // Java's `lats.get(0)` on an empty ring throws; one angle step
        // always lands at most 40 degrees in, so a ring has a vertex.
        lats.push(lats[0]);
        lons.push(lons[0]);
        return Polygon::new(&lats, &lons, vec![]);
    }
}

// Copyright (c) 1970-2003, Wm. Randolph Franklin
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated
// documentation files (the "Software"), to deal in the Software without restriction, including
// without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and
// to permit persons to whom the Software is furnished to do so, subject to the following
// conditions:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimers.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice in the documentation and/or other materials provided with
//    the distribution.
// 3. The name of W. Randolph Franklin may not be used to endorse or
//    promote products derived from this Software without specific
//    prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED
// TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
// NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF
// CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
// OR OTHER DEALINGS
// IN THE SOFTWARE.
/// Java: `GeoTestUtil.containsSlowly`, with inclusive polygon boundaries.
/// Rejects holes before inspecting the bounding box.
pub fn contains_slowly(
    polygon: &Polygon,
    latitude: f64,
    longitude: f64,
) -> Result<bool, &'static str> {
    if !polygon.holes().is_empty() {
        return Err("this testing method does not support holes");
    }
    let x = longitude;
    let y = latitude;
    let ys = polygon.poly_lats();
    let xs = polygon.poly_lons();
    if y < polygon.min_lat || y > polygon.max_lat || x < polygon.min_lon || x > polygon.max_lon {
        return Ok(false);
    }
    let mut c = false;
    for i in 0..xs.len() - 1 {
        let j = i + 1;
        if (y == ys[j] && y == ys[i]) || ((y <= ys[j] && y >= ys[i]) != (y >= ys[j] && y <= ys[i]))
        {
            if (x == xs[j] && x == xs[i])
                || (((x <= xs[j] && x >= xs[i]) != (x >= xs[j] && x <= xs[i]))
                    && GeoUtils::orient(xs[i], ys[i], xs[j], ys[j], x, y) == 0)
            {
                return Ok(true);
            } else if ((ys[i] > y) != (ys[j] > y))
                && x < (xs[j] - xs[i]) * (y - ys[i]) / (ys[j] - ys[i]) + xs[i]
            {
                c = !c;
            }
        }
    }
    Ok(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geographic_ray_crossing_includes_edges_and_rejects_bbox_and_concavity() {
        let polygon = Polygon::new(
            &[0., 6., 6., 2., 2., 0., 0.],
            &[0., 0., 2., 2., 6., 6., 0.],
            vec![],
        )
        .unwrap();
        for (lat, lon, expected) in [
            (1., 1., true),
            (5., 1., true),
            (1., 5., true),
            (5., 5., false),
            (0., 3., true),
            (3., 0., true),
            (2., 6., true),
            (0., -0.0, true),
            (-1., 1., false),
            (7., 1., false),
            (1., -1., false),
            (1., 7., false),
            (f64::NAN, 1., false),
            (1., f64::NAN, false),
        ] {
            assert_eq!(contains_slowly(&polygon, lat, lon), Ok(expected));
        }
    }

    #[test]
    fn java_min_max_follow_math_min_max() {
        assert!(java_min(f64::NAN, 1.0).is_nan() && java_max(1.0, f64::NAN).is_nan());
        assert!(java_min(0.0, -0.0).is_sign_negative());
        assert!(java_min(-0.0, 0.0).is_sign_negative());
        assert!(java_max(-0.0, 0.0).is_sign_positive());
        assert!(java_max(0.0, -0.0).is_sign_positive());
        assert_eq!((java_min(1.0, 2.0), java_min(2.0, 1.0)), (1.0, 1.0));
        assert_eq!((java_max(1.0, 2.0), java_max(2.0, 1.0)), (2.0, 2.0));
    }

    #[test]
    #[should_panic(expected = "invalid latitude 91.0")]
    fn a_near_latitude_needs_a_valid_anchor() {
        next_latitude_near(&mut JavaRandom::new(1), 91.0, 1.0);
    }

    #[test]
    #[should_panic(expected = "invalid longitude 181.0")]
    fn a_near_longitude_needs_a_valid_anchor() {
        next_longitude_near(&mut JavaRandom::new(1), 181.0, 1.0);
    }

    #[test]
    fn a_vertical_or_horizontal_line_targets_its_own_axis() {
        let mut r = JavaRandom::new(3);
        for _ in 0..200 {
            let [lat, lon] = next_point_around_line(&mut r, 0.0, 5.0, 10.0, 5.0);
            assert!((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon));
            let [lat, lon] = next_point_around_line(&mut r, 5.0, 0.0, 5.0, 10.0);
            assert!((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon));
            // A slope steep enough to leave the latitudes is clamped.
            let [lat, _] = next_point_around_line(&mut r, -89.0, 0.0, 89.0, 1e-300);
            assert!((-90.0..=90.0).contains(&lat));
        }
    }

    #[test]
    fn holes_are_refused_before_bbox_or_nonfinite_query_checks() {
        let ring = &[0., 4., 4., 0., 0.];
        let lons = &[0., 0., 4., 4., 0.];
        let hole = Polygon::new(&[1., 2., 2., 1., 1.], &[1., 1., 2., 2., 1.], vec![]).unwrap();
        let polygon = Polygon::new(ring, lons, vec![hole]).unwrap();
        for (lat, lon) in [(1., 1.), (89., 179.), (f64::NAN, f64::INFINITY)] {
            assert_eq!(
                contains_slowly(&polygon, lat, lon),
                Err("this testing method does not support holes")
            );
        }
    }
}
