//! Port of `org.apache.lucene.geo.Component2D`: the relation interface every
//! geo query tests cells, points, lines and triangles against, and its
//! static helpers.
#![allow(clippy::too_many_arguments)]

use super::geo_utils::GeoUtils;
use super::{java_max, java_min, GeoError, Relation};

/// `Component2D.WithinRelation`: used by the `WITHIN` shape query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WithinRelation {
    /// The shape may be within the query (more triangles to check).
    Candidate,
    /// The shape is not within the query.
    NotWithin,
    /// The shape is disjoint from the query.
    Disjoint,
}

/// Port of `org.apache.lucene.geo.Component2D`: a 2D geometry that can
/// relate itself to boxes, points, lines and triangles. `x` is longitude
/// and `y` latitude for lat/lon geometries.
///
/// The `*_bbox` methods are Java's overloads that take the other shape's
/// bounding box first; the plain ones are Java's default methods that
/// compute it.
pub trait Component2D: std::fmt::Debug + Send + Sync {
    /// `getMinX()`.
    fn min_x(&self) -> f64;
    /// `getMaxX()`.
    fn max_x(&self) -> f64;
    /// `getMinY()`.
    fn min_y(&self) -> f64;
    /// `getMaxY()`.
    fn max_y(&self) -> f64;

    /// `contains(x, y)`.
    fn contains(&self, x: f64, y: f64) -> bool;

    /// `relate(minX, maxX, minY, maxY)`: the relation to a box.
    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation;

    /// `intersectsLine(minX, maxX, minY, maxY, aX, aY, bX, bY)`.
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
    ) -> bool;

    /// `intersectsTriangle(minX, maxX, minY, maxY, aX, aY, bX, bY, cX, cY)`.
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
    ) -> bool;

    /// `containsLine(minX, maxX, minY, maxY, aX, aY, bX, bY)`.
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
    ) -> bool;

    /// `containsTriangle(minX, maxX, minY, maxY, aX, aY, bX, bY, cX, cY)`.
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
    ) -> bool;

    /// `withinPoint(x, y)`.
    fn within_point(&self, x: f64, y: f64) -> Result<WithinRelation, GeoError>;

    /// `withinLine(minX, maxX, minY, maxY, aX, aY, ab, bX, bY)`.
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
    ) -> Result<WithinRelation, GeoError>;

    /// `withinTriangle(minX, maxX, minY, maxY, aX, aY, ab, bX, bY, bc, cX, cY, ca)`.
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
    ) -> Result<WithinRelation, GeoError>;

    /// `intersectsLine(aX, aY, bX, bY)`.
    fn intersects_line(&self, a_x: f64, a_y: f64, b_x: f64, b_y: f64) -> bool {
        let min_y = java_min(a_y, b_y);
        let min_x = java_min(a_x, b_x);
        let max_y = java_max(a_y, b_y);
        let max_x = java_max(a_x, b_x);
        self.intersects_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y)
    }

    /// `intersectsTriangle(aX, aY, bX, bY, cX, cY)`.
    fn intersects_triangle(
        &self,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        let min_y = java_min(java_min(a_y, b_y), c_y);
        let min_x = java_min(java_min(a_x, b_x), c_x);
        let max_y = java_max(java_max(a_y, b_y), c_y);
        let max_x = java_max(java_max(a_x, b_x), c_x);
        self.intersects_triangle_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y)
    }

    /// `containsLine(aX, aY, bX, bY)`.
    fn contains_line(&self, a_x: f64, a_y: f64, b_x: f64, b_y: f64) -> bool {
        let min_y = java_min(a_y, b_y);
        let min_x = java_min(a_x, b_x);
        let max_y = java_max(a_y, b_y);
        let max_x = java_max(a_x, b_x);
        self.contains_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y)
    }

    /// `containsTriangle(aX, aY, bX, bY, cX, cY)`.
    fn contains_triangle(
        &self,
        a_x: f64,
        a_y: f64,
        b_x: f64,
        b_y: f64,
        c_x: f64,
        c_y: f64,
    ) -> bool {
        let min_y = java_min(java_min(a_y, b_y), c_y);
        let min_x = java_min(java_min(a_x, b_x), c_x);
        let max_y = java_max(java_max(a_y, b_y), c_y);
        let max_x = java_max(java_max(a_x, b_x), c_x);
        self.contains_triangle_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y)
    }

    /// `withinLine(aX, aY, ab, bX, bY)`.
    fn within_line(
        &self,
        a_x: f64,
        a_y: f64,
        ab: bool,
        b_x: f64,
        b_y: f64,
    ) -> Result<WithinRelation, GeoError> {
        let min_y = java_min(a_y, b_y);
        let min_x = java_min(a_x, b_x);
        let max_y = java_max(a_y, b_y);
        let max_x = java_max(a_x, b_x);
        self.within_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, ab, b_x, b_y)
    }

    /// `withinTriangle(aX, aY, ab, bX, bY, bc, cX, cY, ca)`.
    fn within_triangle(
        &self,
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
        let min_y = java_min(java_min(a_y, b_y), c_y);
        let min_x = java_min(java_min(a_x, b_x), c_x);
        let max_y = java_max(java_max(a_y, b_y), c_y);
        let max_x = java_max(java_max(a_x, b_x), c_x);
        self.within_triangle_bbox(
            min_x, max_x, min_y, max_y, a_x, a_y, ab, b_x, b_y, bc, c_x, c_y, ca,
        )
    }
}

/// `Component2D.disjoint`: the boxes do not intersect.
#[inline]
pub fn disjoint(
    min_x1: f64,
    max_x1: f64,
    min_y1: f64,
    max_y1: f64,
    min_x2: f64,
    max_x2: f64,
    min_y2: f64,
    max_y2: f64,
) -> bool {
    max_y1 < min_y2 || min_y1 > max_y2 || max_x1 < min_x2 || min_x1 > max_x2
}

/// `Component2D.within`: box 1 is within box 2.
#[inline]
pub fn within(
    min_x1: f64,
    max_x1: f64,
    min_y1: f64,
    max_y1: f64,
    min_x2: f64,
    max_x2: f64,
    min_y2: f64,
    max_y2: f64,
) -> bool {
    min_y2 <= min_y1 && max_y2 >= max_y1 && min_x2 <= min_x1 && max_x2 >= max_x1
}

/// `Component2D.containsPoint`: the point is in the box, edges included.
#[inline]
pub fn contains_point(x: f64, y: f64, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
    x >= min_x && x <= max_x && y >= min_y && y <= max_y
}

/// `Component2D.pointInTriangle`: the point is in the triangle (whose
/// bounding box is given), edges included.
#[inline]
pub fn point_in_triangle(
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    x: f64,
    y: f64,
    a_x: f64,
    a_y: f64,
    b_x: f64,
    b_y: f64,
    c_x: f64,
    c_y: f64,
) -> bool {
    // check the bounding box because if the triangle is degenerated, e.g
    // points and lines, we need to filter out coplanar points that are not
    // part of the triangle.
    if x >= min_x && x <= max_x && y >= min_y && y <= max_y {
        let a = GeoUtils::orient(x, y, a_x, a_y, b_x, b_y);
        let b = GeoUtils::orient(x, y, b_x, b_y, c_x, c_y);
        if a == 0 || b == 0 || (a < 0) == (b < 0) {
            let c = GeoUtils::orient(x, y, c_x, c_y, a_x, a_y);
            return c == 0 || ((c < 0) == (b < 0 || a < 0));
        }
        false
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statics() {
        assert!(disjoint(0.0, 1.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0));
        assert!(!disjoint(0.0, 1.0, 0.0, 1.0, 1.0, 3.0, 0.0, 1.0));
        assert!(within(1.0, 2.0, 1.0, 2.0, 0.0, 3.0, 0.0, 3.0));
        assert!(!within(1.0, 4.0, 1.0, 2.0, 0.0, 3.0, 0.0, 3.0));
        assert!(contains_point(1.0, 1.0, 1.0, 1.0, 1.0, 1.0));
        assert!(point_in_triangle(
            0.0, 2.0, 0.0, 2.0, 0.5, 0.5, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0
        ));
        assert!(!point_in_triangle(
            0.0, 2.0, 0.0, 2.0, 1.5, 1.5, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0
        ));
        assert!(!point_in_triangle(
            0.0, 2.0, 0.0, 2.0, 3.0, 3.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0
        ));
        // On an edge.
        assert!(point_in_triangle(
            0.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0
        ));
    }
}
