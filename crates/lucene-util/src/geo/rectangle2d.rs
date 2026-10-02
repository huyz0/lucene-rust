//! Port of `org.apache.lucene.geo.Rectangle2D`: a box as a `Component2D`.
#![allow(clippy::too_many_arguments)]

use super::component2d::{self, Component2D, WithinRelation};
use super::component_tree::ComponentTree;
use super::geo_encoding_utils::GeoEncodingUtils;
use super::geo_utils::GeoUtils;
use super::rectangle::Rectangle;
use super::xy_rectangle::XYRectangle;
use super::{java_double_string, java_max, java_min, GeoError, Relation};

/// Port of `org.apache.lucene.geo.Rectangle2D`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Rectangle2D {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
}

impl Rectangle2D {
    /// `create(XYRectangle)`.
    pub(crate) fn create_xy(rectangle: &XYRectangle) -> Rectangle2D {
        Rectangle2D {
            min_x: f64::from(rectangle.min_x),
            max_x: f64::from(rectangle.max_x),
            min_y: f64::from(rectangle.min_y),
            max_y: f64::from(rectangle.max_y),
        }
    }

    /// `create(Rectangle)`: quantized like `LatLonPoint.newBoxQuery`; a box
    /// crossing the dateline becomes a tree of two.
    pub(crate) fn create(rectangle: &Rectangle) -> Result<Box<dyn Component2D>, GeoError> {
        let mut min_longitude = rectangle.min_lon;
        let mut crosses_dateline = rectangle.min_lon > rectangle.max_lon;
        if min_longitude == 180.0 && crosses_dateline {
            min_longitude = -180.0;
            crosses_dateline = false;
        }
        // need to quantize!
        let q_min_lat = GeoEncodingUtils::decode_latitude(GeoEncodingUtils::encode_latitude_ceil(
            rectangle.min_lat,
        )?);
        let q_max_lat = GeoEncodingUtils::decode_latitude(GeoEncodingUtils::encode_latitude(
            rectangle.max_lat,
        )?);
        let q_min_lon = GeoEncodingUtils::decode_longitude(
            GeoEncodingUtils::encode_longitude_ceil(min_longitude)?,
        );
        let q_max_lon = GeoEncodingUtils::decode_longitude(GeoEncodingUtils::encode_longitude(
            rectangle.max_lon,
        )?);
        if crosses_dateline {
            let min_lon_incl_quantize =
                GeoEncodingUtils::decode_longitude(GeoEncodingUtils::MIN_LON_ENCODED);
            let max_lon_incl_quantize =
                GeoEncodingUtils::decode_longitude(GeoEncodingUtils::MAX_LON_ENCODED);
            let components: Vec<Box<dyn Component2D>> = vec![
                Box::new(Rectangle2D {
                    min_x: min_lon_incl_quantize,
                    max_x: q_max_lon,
                    min_y: q_min_lat,
                    max_y: q_max_lat,
                }),
                Box::new(Rectangle2D {
                    min_x: q_min_lon,
                    max_x: max_lon_incl_quantize,
                    min_y: q_min_lat,
                    max_y: q_max_lat,
                }),
            ];
            Ok(ComponentTree::create(components))
        } else {
            Ok(Box::new(Rectangle2D {
                min_x: q_min_lon,
                max_x: q_max_lon,
                min_y: q_min_lat,
                max_y: q_max_lat,
            }))
        }
    }

    #[inline]
    fn disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        component2d::disjoint(
            self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y,
        )
    }

    /// `edgesIntersect`.
    fn edges_intersect(&self, a_x: f64, a_y: f64, b_x: f64, b_y: f64) -> bool {
        // shortcut: check bboxes of edges are disjoint
        if java_max(a_x, b_x) < self.min_x
            || java_min(a_x, b_x) > self.max_x
            || java_min(a_y, b_y) > self.max_y
            || java_max(a_y, b_y) < self.min_y
        {
            return false;
        }
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        GeoUtils::line_crosses_line_with_boundary(a_x, a_y, b_x, b_y, min_x, max_y, max_x, max_y)
            || GeoUtils::line_crosses_line_with_boundary(
                a_x, a_y, b_x, b_y, max_x, max_y, max_x, min_y,
            )
            || GeoUtils::line_crosses_line_with_boundary(
                a_x, a_y, b_x, b_y, max_x, min_y, min_x, min_y,
            )
            || GeoUtils::line_crosses_line_with_boundary(
                a_x, a_y, b_x, b_y, min_x, min_y, min_x, max_y,
            )
    }
}

impl std::fmt::Display for Rectangle2D {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Rectangle2D(x={} TO {} y={} TO {})",
            java_double_string(self.min_x),
            java_double_string(self.max_x),
            java_double_string(self.min_y),
            java_double_string(self.max_y)
        )
    }
}

impl Component2D for Rectangle2D {
    fn min_x(&self) -> f64 {
        self.min_x
    }
    fn max_x(&self) -> f64 {
        self.max_x
    }
    fn min_y(&self) -> f64 {
        self.min_y
    }
    fn max_y(&self) -> f64 {
        self.max_y
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        component2d::contains_point(x, y, self.min_x, self.max_x, self.min_y, self.max_y)
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Relation::CellOutsideQuery;
        }
        if component2d::within(
            min_x, max_x, min_y, max_y, self.min_x, self.max_x, self.min_y, self.max_y,
        ) {
            return Relation::CellInsideQuery;
        }
        Relation::CellCrossesQuery
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self.edges_intersect(a_x, a_y, b_x, b_y)
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self.contains(c_x, c_y)
            || component2d::point_in_triangle(
                min_x, max_x, min_y, max_y, self.min_x, self.min_y, a_x, a_y, b_x, b_y, c_x, c_y,
            )
            || self.edges_intersect(a_x, a_y, b_x, b_y)
            || self.edges_intersect(b_x, b_y, c_x, c_y)
            || self.edges_intersect(c_x, c_y, a_x, a_y)
    }

    fn contains_line_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
    ) -> bool {
        component2d::within(
            min_x, max_x, min_y, max_y, self.min_x, self.max_x, self.min_y, self.max_y,
        )
    }

    fn contains_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
    ) -> bool {
        component2d::within(
            min_x, max_x, min_y, max_y, self.min_x, self.max_x, self.min_y, self.max_y,
        )
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) {
            return Ok(WithinRelation::NotWithin);
        }
        if ab && self.edges_intersect(a_x, a_y, b_x, b_y) {
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
        // Bounding boxes disjoint?
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        // Points belong to the shape so if points are inside the rectangle then it cannot be within.
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) || self.contains(c_x, c_y) {
            return Ok(WithinRelation::NotWithin);
        }
        // If any of the edges intersects an edge belonging to the shape then it cannot be within.
        let mut relation = WithinRelation::Disjoint;
        for (from_shape, (px, py, qx, qy)) in [
            (ab, (a_x, a_y, b_x, b_y)),
            (bc, (b_x, b_y, c_x, c_y)),
            (ca, (c_x, c_y, a_x, a_y)),
        ] {
            if self.edges_intersect(px, py, qx, qy) {
                if from_shape {
                    return Ok(WithinRelation::NotWithin);
                }
                relation = WithinRelation::Candidate;
            }
        }
        if relation == WithinRelation::Candidate {
            return Ok(WithinRelation::Candidate);
        }
        // Check if shape is within the triangle
        if component2d::point_in_triangle(
            min_x, max_x, min_y, max_y, self.min_x, self.min_y, a_x, a_y, b_x, b_y, c_x, c_y,
        ) {
            return Ok(WithinRelation::Candidate);
        }
        Ok(relation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display() {
        let r = Rectangle2D::create_xy(&XYRectangle::new(-1.0, 1.0, -2.0, 2.0).unwrap());
        assert_eq!(r.to_string(), "Rectangle2D(x=-1.0 TO 1.0 y=-2.0 TO 2.0)");
    }
}
