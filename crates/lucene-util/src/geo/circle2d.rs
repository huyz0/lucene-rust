//! Port of `org.apache.lucene.geo.Circle2D`: a circle (haversine for lat/lon,
//! euclidean for cartesian) as a `Component2D`.
#![allow(clippy::too_many_arguments)]

use super::circle::Circle;
use super::component2d::{self, Component2D, WithinRelation};
use super::geo_utils::GeoUtils;
use super::rectangle::Rectangle;
use super::xy_circle::XYCircle;
use super::xy_rectangle::XYRectangle;
use super::{java_max, java_min, GeoError, Relation};
use crate::sloppy_math;

/// `Circle2D.DistanceCalculator`'s two implementations.
#[derive(Debug, Clone)]
enum DistanceCalculator {
    Cartesian(CartesianDistance),
    Haversin(HaversinDistance),
}

/// `Circle2D.CartesianDistance`.
#[derive(Debug, Clone)]
struct CartesianDistance {
    center_x: f64,
    center_y: f64,
    radius_squared: f64,
    rectangle: XYRectangle,
}

/// `Circle2D.HaversinDistance`.
#[derive(Debug, Clone)]
struct HaversinDistance {
    center_lat: f64,
    center_lon: f64,
    sort_key: f64,
    axis_lat: f64,
    rectangle: Rectangle,
    crosses_dateline: bool,
}

impl CartesianDistance {
    fn new(center_x: f32, center_y: f32, radius: f32) -> Result<CartesianDistance, GeoError> {
        Ok(CartesianDistance {
            center_x: f64::from(center_x),
            center_y: f64::from(center_y),
            rectangle: XYRectangle::from_point_distance(center_x, center_y, radius)?,
            // product performed with doubles
            radius_squared: f64::from(radius) * f64::from(radius),
        })
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if component2d::contains_point(self.center_x, self.center_y, min_x, max_x, min_y, max_y) {
            if self.contains(min_x, min_y)
                && self.contains(max_x, min_y)
                && self.contains(max_x, max_y)
                && self.contains(min_x, max_y)
            {
                // we are fully enclosed, collect everything within this subtree
                return Relation::CellInsideQuery;
            }
        } else {
            // circle not fully inside, compute closest distance
            let mut sum_of_squared_diffs = 0.0f64;
            if self.center_x < min_x {
                let diff = min_x - self.center_x;
                sum_of_squared_diffs += diff * diff;
            } else if self.center_x > max_x {
                let diff = max_x - self.center_x;
                sum_of_squared_diffs += diff * diff;
            }
            if self.center_y < min_y {
                let diff = min_y - self.center_y;
                sum_of_squared_diffs += diff * diff;
            } else if self.center_y > max_y {
                let diff = max_y - self.center_y;
                sum_of_squared_diffs += diff * diff;
            }
            if sum_of_squared_diffs > self.radius_squared {
                // disjoint
                return Relation::CellOutsideQuery;
            }
        }
        Relation::CellCrossesQuery
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        let r = &self.rectangle;
        if component2d::contains_point(
            x,
            y,
            f64::from(r.min_x),
            f64::from(r.max_x),
            f64::from(r.min_y),
            f64::from(r.max_y),
        ) {
            let diff_x = x - self.center_x;
            let diff_y = y - self.center_y;
            return diff_x * diff_x + diff_y * diff_y <= self.radius_squared;
        }
        false
    }

    fn bbox(&self) -> (f64, f64, f64, f64) {
        let r = &self.rectangle;
        (
            f64::from(r.min_x),
            f64::from(r.max_x),
            f64::from(r.min_y),
            f64::from(r.max_y),
        )
    }
}

impl HaversinDistance {
    fn new(center_lon: f64, center_lat: f64, radius: f64) -> Result<HaversinDistance, GeoError> {
        let rectangle = Rectangle::from_point_distance(center_lat, center_lon, radius)?;
        Ok(HaversinDistance {
            center_lat,
            center_lon,
            sort_key: GeoUtils::distance_query_sort_key(radius),
            axis_lat: Rectangle::axis_lat(center_lat, radius),
            crosses_dateline: rectangle.min_lon > rectangle.max_lon,
            rectangle,
        })
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        let r = &self.rectangle;
        let in_box = if self.crosses_dateline {
            component2d::contains_point(
                x,
                y,
                r.min_lon,
                GeoUtils::MAX_LON_INCL,
                r.min_lat,
                r.max_lat,
            ) || component2d::contains_point(
                x,
                y,
                GeoUtils::MIN_LON_INCL,
                r.max_lon,
                r.min_lat,
                r.max_lat,
            )
        } else {
            component2d::contains_point(x, y, r.min_lon, r.max_lon, r.min_lat, r.max_lat)
        };
        if in_box {
            return sloppy_math::haversin_sort_key(y, x, self.center_lat, self.center_lon)
                <= self.sort_key;
        }
        false
    }
}

impl DistanceCalculator {
    fn contains(&self, x: f64, y: f64) -> bool {
        match self {
            DistanceCalculator::Cartesian(c) => c.contains(x, y),
            DistanceCalculator::Haversin(h) => h.contains(x, y),
        }
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        match self {
            DistanceCalculator::Cartesian(c) => c.relate(min_x, max_x, min_y, max_y),
            DistanceCalculator::Haversin(h) => GeoUtils::relate(
                min_y,
                max_y,
                min_x,
                max_x,
                h.center_lat,
                h.center_lon,
                h.sort_key,
                h.axis_lat,
            )
            // Java throws for a dateline-crossing box; Component2D boxes never
            // cross it (min <= max), so this is unreachable in practice.
            .unwrap_or(Relation::CellCrossesQuery),
        }
    }

    fn intersects_line(&self, a_x: f64, a_y: f64, b_x: f64, b_y: f64) -> bool {
        match self {
            DistanceCalculator::Cartesian(c) => {
                intersects_line(c.center_x, c.center_y, a_x, a_y, b_x, b_y, self)
            }
            DistanceCalculator::Haversin(h) => {
                if intersects_line(h.center_lon, h.center_lat, a_x, a_y, b_x, b_y, self) {
                    return true;
                }
                if h.crosses_dateline {
                    let new_center_lon = if h.center_lon > 0.0 {
                        h.center_lon - 360.0
                    } else {
                        h.center_lon + 360.0
                    };
                    return intersects_line(new_center_lon, h.center_lat, a_x, a_y, b_x, b_y, self);
                }
                false
            }
        }
    }

    fn disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        match self {
            DistanceCalculator::Cartesian(c) => {
                let (bx0, bx1, by0, by1) = c.bbox();
                component2d::disjoint(bx0, bx1, by0, by1, min_x, max_x, min_y, max_y)
            }
            DistanceCalculator::Haversin(h) => {
                let r = &h.rectangle;
                if h.crosses_dateline {
                    component2d::disjoint(
                        r.min_lon,
                        GeoUtils::MAX_LON_INCL,
                        r.min_lat,
                        r.max_lat,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    ) && component2d::disjoint(
                        GeoUtils::MIN_LON_INCL,
                        r.max_lon,
                        r.min_lat,
                        r.max_lat,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    )
                } else {
                    component2d::disjoint(
                        r.min_lon, r.max_lon, r.min_lat, r.max_lat, min_x, max_x, min_y, max_y,
                    )
                }
            }
        }
    }

    fn within(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        match self {
            DistanceCalculator::Cartesian(c) => {
                let (bx0, bx1, by0, by1) = c.bbox();
                component2d::within(bx0, bx1, by0, by1, min_x, max_x, min_y, max_y)
            }
            DistanceCalculator::Haversin(h) => {
                let r = &h.rectangle;
                if h.crosses_dateline {
                    component2d::within(
                        r.min_lon,
                        GeoUtils::MAX_LON_INCL,
                        r.min_lat,
                        r.max_lat,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    ) || component2d::within(
                        GeoUtils::MIN_LON_INCL,
                        r.max_lon,
                        r.min_lat,
                        r.max_lat,
                        min_x,
                        max_x,
                        min_y,
                        max_y,
                    )
                } else {
                    component2d::within(
                        r.min_lon, r.max_lon, r.min_lat, r.max_lat, min_x, max_x, min_y, max_y,
                    )
                }
            }
        }
    }

    fn min_x(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => f64::from(c.rectangle.min_x),
            DistanceCalculator::Haversin(h) => {
                if h.crosses_dateline {
                    GeoUtils::MIN_LON_INCL
                } else {
                    h.rectangle.min_lon
                }
            }
        }
    }

    fn max_x(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => f64::from(c.rectangle.max_x),
            DistanceCalculator::Haversin(h) => {
                if h.crosses_dateline {
                    GeoUtils::MAX_LON_INCL
                } else {
                    h.rectangle.max_lon
                }
            }
        }
    }

    fn min_y(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => f64::from(c.rectangle.min_y),
            DistanceCalculator::Haversin(h) => h.rectangle.min_lat,
        }
    }

    fn max_y(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => f64::from(c.rectangle.max_y),
            DistanceCalculator::Haversin(h) => h.rectangle.max_lat,
        }
    }

    fn x(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => c.center_x,
            DistanceCalculator::Haversin(h) => h.center_lon,
        }
    }

    fn y(&self) -> f64 {
        match self {
            DistanceCalculator::Cartesian(c) => c.center_y,
            DistanceCalculator::Haversin(h) => h.center_lat,
        }
    }
}

/// `Circle2D.intersectsLine`: whether the closest point of the segment to
/// the centre is inside the circle.
fn intersects_line(
    center_x: f64,
    center_y: f64,
    a_x: f64,
    a_y: f64,
    b_x: f64,
    b_y: f64,
    calculator: &DistanceCalculator,
) -> bool {
    let vector_apx = center_x - a_x;
    let vector_apy = center_y - a_y;
    let vector_abx = b_x - a_x;
    let vector_aby = b_y - a_y;
    let magnitude_ab = vector_abx * vector_abx + vector_aby * vector_aby;
    let dot_product = vector_apx * vector_abx + vector_apy * vector_aby;
    let distance = dot_product / magnitude_ab;
    // Java's test: a NaN distance falls through (`contains` would not).
    #[allow(clippy::manual_range_contains)]
    if distance < 0.0 || distance > 1.0 {
        return false;
    }
    let p_x = a_x + vector_abx * distance;
    let p_y = a_y + vector_aby * distance;
    let min_x = java_min(a_x, b_x);
    let min_y = java_min(a_y, b_y);
    let max_x = java_max(a_x, b_x);
    let max_y = java_max(a_y, b_y);
    if p_x >= min_x && p_x <= max_x && p_y >= min_y && p_y <= max_y {
        return calculator.contains(p_x, p_y);
    }
    false
}

/// Port of `org.apache.lucene.geo.Circle2D`.
#[derive(Debug, Clone)]
pub(crate) struct Circle2D {
    calculator: DistanceCalculator,
}

impl Circle2D {
    /// `create(XYCircle)`.
    pub(crate) fn create_xy(circle: &XYCircle) -> Result<Circle2D, GeoError> {
        Ok(Circle2D {
            calculator: DistanceCalculator::Cartesian(CartesianDistance::new(
                circle.x(),
                circle.y(),
                circle.radius(),
            )?),
        })
    }

    /// `create(Circle)`.
    pub(crate) fn create(circle: &Circle) -> Result<Circle2D, GeoError> {
        Ok(Circle2D {
            calculator: DistanceCalculator::Haversin(HaversinDistance::new(
                circle.lon(),
                circle.lat(),
                circle.radius(),
            )?),
        })
    }
}

impl Component2D for Circle2D {
    fn min_x(&self) -> f64 {
        self.calculator.min_x()
    }
    fn max_x(&self) -> f64 {
        self.calculator.max_x()
    }
    fn min_y(&self) -> f64 {
        self.calculator.min_y()
    }
    fn max_y(&self) -> f64 {
        self.calculator.max_y()
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        self.calculator.contains(x, y)
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return Relation::CellOutsideQuery;
        }
        if self.calculator.within(min_x, max_x, min_y, max_y) {
            return Relation::CellCrossesQuery;
        }
        self.calculator.relate(min_x, max_x, min_y, max_y)
    }

    fn intersects_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
    ) -> bool {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self.calculator.intersects_line(a_x, a_y, b_x, b_y)
    }

    fn intersects_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self.contains(c_x, c_y)
            || component2d::point_in_triangle(
                min_x,
                max_x,
                min_y,
                max_y,
                self.calculator.x(),
                self.calculator.y(),
                a_x,
                a_y,
                b_x,
                b_y,
                c_x,
                c_y,
            )
            || self.calculator.intersects_line(a_x, a_y, b_x, b_y)
            || self.calculator.intersects_line(b_x, b_y, c_x, c_y)
            || self.calculator.intersects_line(c_x, c_y, a_x, a_y)
    }

    fn contains_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
    ) -> bool {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y) && self.contains(b_x, b_y)
    }

    fn contains_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y) && self.contains(b_x, b_y) && self.contains(c_x, c_y)
    }

    fn within_point(&self, x: f64, y: f64) -> Result<WithinRelation, GeoError> {
        Ok(if self.contains(x, y) {
            WithinRelation::NotWithin
        } else {
            WithinRelation::Disjoint
        })
    }

    fn within_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        ab: bool,
        b_x: f64,
        b_y: f64,
    ) -> Result<WithinRelation, GeoError> {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) {
            return Ok(WithinRelation::NotWithin);
        }
        if ab && self.calculator.intersects_line(a_x, a_y, b_x, b_y) {
            return Ok(WithinRelation::NotWithin);
        }
        Ok(WithinRelation::Disjoint)
    }

    fn within_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        ab: bool,
        b_x: f64,
        b_y: f64,
        bc: bool,
        c_x: f64,
        c_y: f64,
        ca: bool,
    ) -> Result<WithinRelation, GeoError> {
        if self.calculator.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        // if any of the points is inside the circle then we cannot be within this
        // indexed shape
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) || self.contains(c_x, c_y) {
            return Ok(WithinRelation::NotWithin);
        }
        // we only check edges that belong to the original polygon. If we intersect any of them, then
        // we are not within.
        if ab && self.calculator.intersects_line(a_x, a_y, b_x, b_y) {
            return Ok(WithinRelation::NotWithin);
        }
        if bc && self.calculator.intersects_line(b_x, b_y, c_x, c_y) {
            return Ok(WithinRelation::NotWithin);
        }
        if ca && self.calculator.intersects_line(c_x, c_y, a_x, a_y) {
            return Ok(WithinRelation::NotWithin);
        }
        // check if center is within the triangle.
        if component2d::point_in_triangle(
            min_x,
            max_x,
            min_y,
            max_y,
            self.calculator.x(),
            self.calculator.y(),
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
        ) {
            return Ok(WithinRelation::Candidate);
        }
        Ok(WithinRelation::Disjoint)
    }
}
