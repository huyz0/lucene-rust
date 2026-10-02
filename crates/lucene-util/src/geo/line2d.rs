//! Port of `org.apache.lucene.geo.Line2D`: a polyline as a `Component2D`.
#![allow(clippy::too_many_arguments)]

use super::component2d::{self, Component2D, WithinRelation};
use super::edge_tree::EdgeTree;
use super::line::Line;
use super::xy_encoding_utils::XYEncodingUtils;
use super::xy_line::XYLine;
use super::{GeoError, Relation};

/// Port of `org.apache.lucene.geo.Line2D`.
#[derive(Debug, Clone)]
pub(crate) struct Line2D {
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
    tree: EdgeTree,
}

impl Line2D {
    /// `create(Line)`.
    pub(crate) fn create(line: &Line) -> Line2D {
        Line2D {
            min_y: line.min_lat,
            max_y: line.max_lat,
            min_x: line.min_lon,
            max_x: line.max_lon,
            tree: EdgeTree::create_tree(line.lons(), line.lats()),
        }
    }

    /// `create(XYLine)`.
    pub(crate) fn create_xy(line: &XYLine) -> Line2D {
        Line2D {
            min_y: f64::from(line.min_y),
            max_y: f64::from(line.max_y),
            min_x: f64::from(line.min_x),
            max_x: f64::from(line.max_x),
            tree: EdgeTree::create_tree(
                &XYEncodingUtils::float_array_to_double_array(line.x()),
                &XYEncodingUtils::float_array_to_double_array(line.y()),
            ),
        }
    }

    #[inline]
    fn disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        component2d::disjoint(
            self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y,
        )
    }
}

impl Component2D for Line2D {
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
        if component2d::contains_point(x, y, self.min_x, self.max_x, self.min_y, self.max_y) {
            return self.tree.is_point_on_line(x, y);
        }
        false
    }

    fn relate(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> Relation {
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Relation::CellOutsideQuery;
        }
        if component2d::within(
            self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y,
        ) {
            return Relation::CellCrossesQuery;
        }
        if self.tree.crosses_box(min_x, max_x, min_y, max_y, true) {
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        self.tree
            .crosses_line(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, true)
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
        component2d::point_in_triangle(
            min_x,
            max_x,
            min_y,
            max_y,
            self.tree.x1,
            self.tree.y1,
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
        ) || self.tree.crosses_triangle(
            min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, true,
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
        if ab && self.intersects_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y) {
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        let mut relation = WithinRelation::Disjoint;
        // if any of the edges intersects an the edge belongs to the shape then it cannot be within.
        // if it only intersects edges that do not belong to the shape, then it is a candidate
        // we skip edges at the dateline to support shapes crossing it
        for (from_shape, (px, py, qx, qy)) in [
            (ab, (a_x, a_y, b_x, b_y)),
            (bc, (b_x, b_y, c_x, c_y)),
            (ca, (c_x, c_y, a_x, a_y)),
        ] {
            if self
                .tree
                .crosses_line(min_x, max_x, min_y, max_y, px, py, qx, qy, true)
            {
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
            min_x,
            max_x,
            min_y,
            max_y,
            self.tree.x1,
            self.tree.y1,
            a_x,
            a_y,
            b_x,
            b_y,
            c_x,
            c_y,
        ) {
            return Ok(WithinRelation::Candidate);
        }
        Ok(relation)
    }
}
