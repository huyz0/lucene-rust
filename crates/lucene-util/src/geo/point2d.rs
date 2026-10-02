//! Port of `org.apache.lucene.geo.Point2D`: a point as a `Component2D`.
#![allow(clippy::too_many_arguments)]

use super::component2d::{self, Component2D, WithinRelation};
use super::geo_encoding_utils::GeoEncodingUtils;
use super::geo_utils::GeoUtils;
use super::point::Point;
use super::xy_point::XYPoint;
use super::{GeoError, Relation};

/// Port of `org.apache.lucene.geo.Point2D`.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Point2D {
    x: f64,
    y: f64,
}

impl Point2D {
    /// `create(Point)`: the point quantized up, as `LatLonPoint` would index
    /// it (points behave as rectangles).
    pub(crate) fn create(point: &Point) -> Result<Point2D, GeoError> {
        let q_lat = if point.lat() == GeoUtils::MAX_LAT_INCL {
            point.lat()
        } else {
            GeoEncodingUtils::decode_latitude(GeoEncodingUtils::encode_latitude_ceil(point.lat())?)
        };
        let q_lon = if point.lon() == GeoUtils::MAX_LON_INCL {
            point.lon()
        } else {
            GeoEncodingUtils::decode_longitude(GeoEncodingUtils::encode_longitude_ceil(
                point.lon(),
            )?)
        };
        Ok(Point2D { x: q_lon, y: q_lat })
    }

    /// `create(XYPoint)`.
    pub(crate) fn create_xy(point: &XYPoint) -> Point2D {
        Point2D {
            x: f64::from(point.x()),
            y: f64::from(point.y()),
        }
    }
}

impl Component2D for Point2D {
    fn min_x(&self) -> f64 {
        self.x
    }
    fn max_x(&self) -> f64 {
        self.x
    }
    fn min_y(&self) -> f64 {
        self.y
    }
    fn max_y(&self) -> f64 {
        self.y
    }

    fn contains(&self, x: f64, y: f64) -> bool {
        x == self.x && y == self.y
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if component2d::contains_point(self.x, self.y, min_x, max_x, min_y, max_y) {
            return Relation::CellCrossesQuery;
        }
        Relation::CellOutsideQuery
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
        component2d::contains_point(self.x, self.y, min_x, max_x, min_y, max_y)
            && GeoUtils::orient(a_x, a_y, b_x, b_y, self.x, self.y) == 0
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
        component2d::point_in_triangle(
            min_x, max_x, min_y, max_y, self.x, self.y, a_x, a_y, b_x, b_y, c_x, c_y,
        )
    }

    fn contains_line_bbox(
        &self,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
    ) -> bool {
        false
    }

    fn contains_triangle_bbox(
        &self,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
        _: f64,
    ) -> bool {
        false
    }

    fn within_point(&self, x: f64, y: f64) -> Result<WithinRelation, GeoError> {
        Ok(if self.contains(x, y) {
            WithinRelation::Candidate
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
        _ab: bool,
        b_x: f64,
        b_y: f64,
    ) -> Result<WithinRelation, GeoError> {
        Ok(
            if self.intersects_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y) {
                WithinRelation::Candidate
            } else {
                WithinRelation::Disjoint
            },
        )
    }

    fn within_triangle_bbox(
        &self,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        a_x: f64,
        a_y: f64,
        _ab: bool,
        b_x: f64,
        b_y: f64,
        _bc: bool,
        c_x: f64,
        c_y: f64,
        _ca: bool,
    ) -> Result<WithinRelation, GeoError> {
        Ok(
            if component2d::point_in_triangle(
                min_x, max_x, min_y, max_y, self.x, self.y, a_x, a_y, b_x, b_y, c_x, c_y,
            ) {
                WithinRelation::Candidate
            } else {
                WithinRelation::Disjoint
            },
        )
    }
}
