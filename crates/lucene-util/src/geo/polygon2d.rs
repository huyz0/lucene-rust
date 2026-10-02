//! Port of `org.apache.lucene.geo.Polygon2D`: a polygon (with holes) as a
//! `Component2D`.
#![allow(clippy::too_many_arguments)]

use super::component2d::{self, Component2D, WithinRelation};
use super::edge_tree::EdgeTree;
use super::lat_lon_geometry::LatLonGeometry;
use super::polygon::Polygon;
use super::xy_encoding_utils::XYEncodingUtils;
use super::xy_geometry::XYGeometry;
use super::xy_polygon::XYPolygon;
use super::{GeoError, Relation};

/// Port of `org.apache.lucene.geo.Polygon2D`.
#[derive(Debug)]
pub(crate) struct Polygon2D {
    min_y: f64,
    max_y: f64,
    min_x: f64,
    max_x: f64,
    holes: Option<Box<dyn Component2D>>,
    tree: EdgeTree,
}

impl Polygon2D {
    /// `create(Polygon)`.
    pub(crate) fn create(polygon: &Polygon) -> Result<Polygon2D, GeoError> {
        let holes = if polygon.num_holes() > 0 {
            let geoms: Vec<LatLonGeometry> = polygon
                .holes()
                .iter()
                .cloned()
                .map(LatLonGeometry::Polygon)
                .collect();
            Some(LatLonGeometry::create(&geoms)?)
        } else {
            None
        };
        Ok(Polygon2D {
            min_y: polygon.min_lat,
            max_y: polygon.max_lat,
            min_x: polygon.min_lon,
            max_x: polygon.max_lon,
            holes,
            tree: EdgeTree::create_tree(polygon.poly_lons(), polygon.poly_lats()),
        })
    }

    /// `create(XYPolygon)`.
    pub(crate) fn create_xy(polygon: &XYPolygon) -> Result<Polygon2D, GeoError> {
        let holes = if polygon.num_holes() > 0 {
            let geoms: Vec<XYGeometry> = polygon
                .holes()
                .iter()
                .cloned()
                .map(XYGeometry::Polygon)
                .collect();
            Some(XYGeometry::create(&geoms)?)
        } else {
            None
        };
        Ok(Polygon2D {
            min_y: f64::from(polygon.min_y),
            max_y: f64::from(polygon.max_y),
            min_x: f64::from(polygon.min_x),
            max_x: f64::from(polygon.max_x),
            holes,
            tree: EdgeTree::create_tree(
                &XYEncodingUtils::float_array_to_double_array(polygon.poly_x()),
                &XYEncodingUtils::float_array_to_double_array(polygon.poly_y()),
            ),
        })
    }

    #[inline]
    fn disjoint(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> bool {
        component2d::disjoint(
            self.min_x, self.max_x, self.min_y, self.max_y, min_x, max_x, min_y, max_y,
        )
    }

    /// `numberOfCorners`: 0, 4, or something in between.
    fn number_of_corners(&self, min_x: f64, max_x: f64, min_y: f64, max_y: f64) -> i32 {
        let mut contains_count = 0;
        if self.contains(min_x, min_y) {
            contains_count += 1;
        }
        if self.contains(max_x, min_y) {
            contains_count += 1;
        }
        if contains_count == 1 {
            return contains_count;
        }
        if self.contains(max_x, max_y) {
            contains_count += 1;
        }
        if contains_count == 2 {
            return contains_count;
        }
        if self.contains(min_x, max_y) {
            contains_count += 1;
        }
        contains_count
    }
}

impl Component2D for Polygon2D {
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
        if component2d::contains_point(x, y, self.min_x, self.max_x, self.min_y, self.max_y)
            && self.tree.contains(x, y)
        {
            return match &self.holes {
                None => true,
                Some(h) => !h.contains(x, y),
            };
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
        // check any holes
        if let Some(holes) = &self.holes {
            match holes.relate(min_x, max_x, min_y, max_y) {
                Relation::CellCrossesQuery => return Relation::CellCrossesQuery,
                Relation::CellInsideQuery => return Relation::CellOutsideQuery,
                Relation::CellOutsideQuery => {}
            }
        }
        // check each corner: if < 4 && > 0 are present, its cheaper than crossesSlowly
        let num_corners = self.number_of_corners(min_x, max_x, min_y, max_y);
        if num_corners == 4 {
            if self.tree.crosses_box(min_x, max_x, min_y, max_y, true) {
                return Relation::CellCrossesQuery;
            }
            return Relation::CellInsideQuery;
        } else if num_corners == 0 {
            if component2d::contains_point(self.tree.x1, self.tree.y1, min_x, max_x, min_y, max_y) {
                return Relation::CellCrossesQuery;
            }
            if self.tree.crosses_box(min_x, max_x, min_y, max_y, true) {
                return Relation::CellCrossesQuery;
            }
            return Relation::CellOutsideQuery;
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
        if self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self
                .tree
                .crosses_line(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, true)
        {
            return match &self.holes {
                None => true,
                Some(h) => !h.contains_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y),
            };
        }
        false
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
        if self.contains(a_x, a_y)
            || self.contains(b_x, b_y)
            || self.contains(c_x, c_y)
            || component2d::point_in_triangle(
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
            )
            || self.tree.crosses_triangle(
                min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, true,
            )
        {
            return match &self.holes {
                None => true,
                Some(h) => !h.contains_triangle_bbox(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y,
                ),
            };
        }
        false
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        if self.contains(a_x, a_y)
            && self.contains(b_x, b_y)
            && !self
                .tree
                .crosses_line(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, false)
        {
            return match &self.holes {
                None => true,
                Some(h) => !h.intersects_line_bbox(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y),
            };
        }
        false
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return false;
        }
        if self.contains(a_x, a_y)
            && self.contains(b_x, b_y)
            && self.contains(c_x, c_y)
            && !self.tree.crosses_triangle(
                min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y, false,
            )
        {
            return match &self.holes {
                None => true,
                Some(h) => !h.intersects_triangle_bbox(
                    min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, c_x, c_y,
                ),
            };
        }
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
        if self.disjoint(min_x, max_x, min_y, max_y) {
            return Ok(WithinRelation::Disjoint);
        }
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) {
            return Ok(WithinRelation::NotWithin);
        }
        if ab
            && self
                .tree
                .crosses_line(min_x, max_x, min_y, max_y, a_x, a_y, b_x, b_y, true)
        {
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
        // if any of the points is inside the polygon, the polygon cannot be within this indexed
        // shape because points belong to the original indexed shape.
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) || self.contains(c_x, c_y) {
            return Ok(WithinRelation::NotWithin);
        }
        let mut relation = WithinRelation::Disjoint;
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
