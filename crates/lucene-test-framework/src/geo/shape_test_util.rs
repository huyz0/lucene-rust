//! `ShapeTestUtil`: biased cartesian coordinates, random shapes and a slow
//! independent polygon oracle. RandomizedContext is an explicit JavaRandom;
//! TEST_NIGHTLY is an explicit flag. Geometry constructors retain validation.
//! Float arithmetic, RNG draw order and Java's regular-polygon coordinate swap
//! are intentional. BiasedNumbers is randomizedtesting 2.8.1, not a uniform draw.
//! Side counts use usize: Java's negative counts and signed-int upper bound
//! are outside this API's validation domain.

use lucene_util::geo::{GeoError, GeoUtils, XYCircle, XYLine, XYPoint, XYPolygon, XYRectangle};
use lucene_util::java_random::JavaRandom;
use lucene_util::strict_math;

use crate::util::random_numbers::random_int_between;

/// Java: `ShapeTestUtil.nextFloat`, over the entire finite float domain.
pub fn next_float(random: &mut JavaRandom) -> f32 {
    let pick = random.next_int_bounded(47);
    biased_float(random, pick)
}

// Java: BiasedNumbers.randomFloatBetween for [-Float.MAX_VALUE, Float.MAX_VALUE].
fn biased_float(random: &mut JavaRandom, pick: i32) -> f32 {
    match pick {
        0 => -f32::MAX,
        1 => f32::MAX,
        2..=21 => {
            let up = random.next_boolean();
            fuzz(random, if up { -f32::MAX } else { f32::MAX }, up)
        }
        22..=26 => match random.next_int_bounded(4) {
            0 => 0.0,
            1 => -0.0,
            2 => fuzz(random, 0.0, false),
            _ => fuzz(random, 0.0, true),
        },
        27..=36 => {
            (f64::from(-f32::MAX)
                + (f64::from(f32::MAX) - f64::from(-f32::MAX)) * random.next_double())
                as f32
        }
        37..=46 => {
            let bits = random_int_between(
                random,
                flip((-f32::MAX).to_bits() as i32),
                flip(f32::MAX.to_bits() as i32),
            );
            f32::from_bits(flip(bits) as u32)
        }
        _ => unreachable!("a biased pick is in 0..47"),
    }
}
fn flip(bits: i32) -> i32 {
    bits ^ ((bits >> 31) & i32::MAX)
}
fn fuzz(random: &mut JavaRandom, mut value: f32, up: bool) -> f32 {
    for _ in 0..random_int_between(random, 1, 10) {
        value = if up {
            value.next_up()
        } else {
            value.next_down()
        };
    }
    value
}

/// Java: `ShapeTestUtil.nextBox`: both sides are nonzero, including ±0 equality.
pub fn next_box(random: &mut JavaRandom) -> XYRectangle {
    let mut x0 = next_float(random);
    let mut x1 = next_float(random);
    while x0 == x1 {
        x1 = next_float(random);
    }
    let mut y0 = next_float(random);
    let mut y1 = next_float(random);
    while y0 == y1 {
        y1 = next_float(random);
    }
    if x1 < x0 {
        std::mem::swap(&mut x0, &mut x1);
    }
    if y1 < y0 {
        std::mem::swap(&mut y0, &mut y1);
    }
    XYRectangle::new(x0, x1, y0, y1).expect("finite ordered coordinates")
}

/// Java: `ShapeTestUtil.nextXYPoint`.
pub fn next_xy_point(random: &mut JavaRandom) -> XYPoint {
    let x = next_float(random);
    let y = next_float(random);
    XYPoint::new(x, y).expect("finite coordinates")
}

/// Java: `ShapeTestUtil.nextCircle`; multiplication and division occur in float.
pub fn next_circle(random: &mut JavaRandom) -> XYCircle {
    let x = next_float(random);
    let y = next_float(random);
    let mut radius = 0.0;
    while radius == 0.0 {
        radius = random.next_float() * f32::MAX / 2.0;
    }
    XYCircle::new(x, y, radius).expect("positive finite radius")
}

fn box_polygon(b: XYRectangle, triangle: bool) -> Result<XYPolygon, GeoError> {
    if triangle {
        XYPolygon::new(
            &[b.min_x, b.max_x, b.max_x, b.min_x],
            &[b.min_y, b.min_y, b.max_y, b.min_y],
            vec![],
        )
    } else {
        XYPolygon::new(
            &[b.min_x, b.max_x, b.max_x, b.min_x, b.min_x],
            &[b.min_y, b.min_y, b.max_y, b.max_y, b.min_y],
            vec![],
        )
    }
}

/// Java: `ShapeTestUtil.nextPolygon`. The nightly-only draw is skipped entirely
/// when `nightly` is false; failed regular-polygon validation retries.
pub fn next_polygon(random: &mut JavaRandom, nightly: bool) -> Result<XYPolygon, GeoError> {
    if random.next_boolean() {
        surprise_polygon(random)
    } else if nightly && random.next_int_bounded(10) == 1 {
        loop {
            let gons = random_int_between(random, 4, 500) as usize;
            let radius = random.next_double() * 0.5 * f64::from(f32::MAX) + 1.0;
            let x = next_float(random);
            let y = next_float(random);
            if let Ok(polygon) = create_regular_polygon(f64::from(x), f64::from(y), radius, gons) {
                return Ok(polygon);
            }
        }
    } else {
        let b = next_box(random);
        box_polygon(b, !random.next_boolean())
    }
}

fn surprise_polygon(random: &mut JavaRandom) -> Result<XYPolygon, GeoError> {
    let center_x = next_float(random);
    let center_y = next_float(random);
    let radius = 0.1 + 20.0 * random.next_double();
    let delta = random.next_double();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut angle = 0.0;
    loop {
        angle += random.next_double() * 40.0;
        if angle > 360.0 {
            break;
        }
        let mut len = radius * (1.0 - delta + delta * random.next_double());
        // The subtraction and abs are float in this method (double below).
        let max_x = (f32::MAX - center_x)
            .abs()
            .min((-f32::MAX - center_x).abs());
        let max_y = (f32::MAX - center_y)
            .abs()
            .min((-f32::MAX - center_y).abs());
        len = len.min(f64::from(max_x.min(max_y)));
        let radians = angle.to_radians();
        xs.push((f64::from(center_x) + len * strict_math::cos(radians)) as f32);
        ys.push((f64::from(center_y) + len * strict_math::sin(radians)) as f32);
    }
    xs.push(xs[0]);
    ys.push(ys[0]);
    XYPolygon::new(&xs, &ys, vec![])
}

/// Java: `ShapeTestUtil.nextLine`, excluding the polygon's closure vertex.
pub fn next_line(random: &mut JavaRandom, nightly: bool) -> Result<XYLine, GeoError> {
    let polygon = next_polygon(random, nightly)?;
    let end = polygon.num_points() - 1;
    XYLine::new(&polygon.poly_x()[..end], &polygon.poly_y()[..end])
}

/// Java: `ShapeTestUtil.createRegularPolygon`; the source places centerY in
/// the X array and centerX in the Y array. At least three sides are required.
pub fn create_regular_polygon(
    center_x: f64,
    center_y: f64,
    mut radius: f64,
    gons: usize,
) -> Result<XYPolygon, GeoError> {
    let max = f64::from(f32::MAX);
    let max_x = (max - center_x).abs().min((-max - center_x).abs());
    let max_y = (max - center_y).abs().min((-max - center_y).abs());
    radius = java_min(radius, java_min(max_x, max_y));
    let mut xs = Vec::with_capacity(gons.saturating_add(1));
    let mut ys = Vec::with_capacity(gons.saturating_add(1));
    for i in 0..gons {
        let angle = 360.0 - i as f64 * (360.0 / gons as f64);
        let radians = angle.to_radians();
        xs.push((center_y + strict_math::sin(radians) * radius) as f32);
        ys.push((center_x + strict_math::cos(radians) * radius) as f32);
    }
    if gons == 0 {
        return Err(GeoError::IllegalArgument(
            "at least 4 polygon points required".into(),
        ));
    }
    xs.push(xs[0]);
    ys.push(ys[0]);
    XYPolygon::new(&xs, &ys, vec![])
}
fn java_min(a: f64, b: f64) -> f64 {
    // Rust min permits either signed zero; Java requires negative zero.
    if a.is_nan() {
        return a;
    }
    if a == 0.0 && b == 0.0 && b.is_sign_negative() {
        return b;
    }
    if a <= b {
        a
    } else {
        b
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
/// Java: `ShapeTestUtil.containsSlowly`, with inclusive polygon boundaries.
/// Rejects holes before inspecting the bounding box.
pub fn contains_slowly(polygon: &XYPolygon, x: f64, y: f64) -> Result<bool, &'static str> {
    if !polygon.holes().is_empty() {
        return Err("this testing method does not support holes");
    }
    let xs: Vec<f64> = polygon.poly_x().iter().map(|&v| f64::from(v)).collect();
    let ys: Vec<f64> = polygon.poly_y().iter().map(|&v| f64::from(v)).collect();
    if x < f64::from(polygon.min_x)
        || x > f64::from(polygon.max_x)
        || y < f64::from(polygon.min_y)
        || y > f64::from(polygon.max_y)
    {
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
    fn actual_java_scripted_biased_picks_preserve_bits_and_continuation() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/shape_test_util_scripted.tsv");
        let fixture = std::fs::read_to_string(path).unwrap();
        for line in fixture.lines() {
            let fields: Vec<_> = line.split('\t').collect();
            let pick = fields[0].parse().unwrap();
            let seed = fields[1].parse().unwrap();
            let mut random = JavaRandom::new(seed);
            assert_eq!(
                biased_float(&mut random, pick).to_bits().to_string(),
                fields[2],
                "{line}"
            );
            assert_eq!(random.next_int().to_string(), fields[3], "{line}");
        }
    }
    #[test]
    fn regular_polygon_rejects_empty_sides_and_nonfinite_coordinates() {
        assert!(create_regular_polygon(0.0, 0.0, 1.0, 0).is_err());
        assert!(create_regular_polygon(f64::NAN, 0.0, 1.0, 4).is_err());
        assert!(create_regular_polygon(0.0, 0.0, f64::NAN, 4).is_err());
        assert!(create_regular_polygon(f64::INFINITY, 0.0, 1.0, 4).is_err());
    }
    #[test]
    fn java_min_keeps_negative_zero_in_either_argument_and_nan_payloads() {
        for (a, b, expected) in [
            (0.0, -0.0, -0.0),
            (-0.0, 0.0, -0.0),
            (-0.0, -0.0, -0.0),
            (0.0, 0.0, 0.0),
        ] {
            assert_eq!(java_min(a, b).to_bits(), f64::to_bits(expected));
        }
        let nan = f64::from_bits(0x7ff8_0000_0000_0042);
        assert_eq!(java_min(nan, 1.0).to_bits(), nan.to_bits());
        assert_eq!(java_min(1.0, nan).to_bits(), nan.to_bits());
    }
}
