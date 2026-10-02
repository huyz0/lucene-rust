//! The shape queries: `LatLonShapeQuery`, `LatLonShapeBoundingBoxQuery` and
//! `XYShapeQuery` (the BKD-backed half of `SpatialQuery`, over the seven
//! dimension triangles `ShapeField` encodes), and the doc-values forms
//! `LatLonShapeDocValuesQuery` / `XYShapeDocValuesQuery`
//! (`BaseShapeDocValuesQuery`, over the `ShapeDocValues` tree).
//!
//! The tree walk is `SpatialQuery`'s, shared with `LatLonPointQuery`
//! ([`spatial_score_leaf`]): the cell relation here reads a cell's bounds
//! as the four indexed dimensions -- the triangles' `minY, minX, maxY,
//! maxX` -- and a leaf value is decoded (`ShapeField.decodeTriangle`) and
//! related as the point, line or triangle it is.
//!
//! # Deviations
//!
//! - `LatLonShapeBoundingBoxQuery` compares a cell's bounds with the encoded
//!   box `int` by `int` where Java compares the sortable bytes
//!   (`ArrayUtil.compareUnsigned4`); the two orders are the same.
//! - A field whose tree is not seven four-byte dimensions, four indexed, is
//!   an error before the walk (Java would throw decoding a short value).

use std::borrow::Cow;
use std::sync::Arc;

use lucene_codecs::doc_values::BinaryReader;
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::points::{PointsField, Relation};
use lucene_index::document::{
    sortable_bytes_to_int, ShapeDocValues, ShapeEncoding, ShapeField, TriangleType,
    XYShapeDocValues, TRIANGLE_BYTES,
};
use lucene_util::geo::{
    point_in_triangle, Component2D, GeoEncodingUtils, GeoError, GeoUtils, LatLonGeometry,
    Rectangle, WithinRelation, XYGeometry,
};

use super::point_queries::{points_of, spatial_score_leaf, SpatialVisitor};
use super::{geo, illegal, QueryRelation};
use crate::collector::ScoringCollector;
use crate::document::{collect_live, field_info, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// The `int` at byte `offset` of a packed value.
#[inline]
fn int_at(packed: &[u8], offset: usize) -> i32 {
    sortable_bytes_to_int(&packed[offset..offset + ShapeField::BYTES])
}

/// A leaf value as an encoded triangle. The tree was checked to hold seven
/// four-byte dimensions ([`check_shape_tree`]), so every value is one.
#[inline]
fn triangle(packed: &[u8]) -> Option<&[u8; TRIANGLE_BYTES]> {
    packed.try_into().ok()
}

/// The tree must hold `ShapeField`'s seven four-byte dimensions, four of
/// them indexed: every visitor here reads a cell's bounds at bytes 0..16
/// and decodes a value's 28 bytes.
fn check_shape_tree(name: &str, pf: &PointsField) -> Result<()> {
    if pf.num_dims != 7 || pf.num_index_dims != 4 || pf.bytes_per_dim != 4 {
        return Err(illegal(format!(
            "field=\"{name}\" holds points of {} dimensions ({} indexed) of {} bytes, not a \
             shape's seven of four (four indexed)",
            pf.num_dims, pf.num_index_dims, pf.bytes_per_dim
        )));
    }
    Ok(())
}

/// `SpatialQuery.getScorerSupplier` over a shape field: the walk, once the
/// tree's shape is checked.
fn score_shape_leaf<V: SpatialVisitor + ?Sized>(
    field: &str,
    visitor: &V,
    rel: QueryRelation,
    leaf: &OpenSegment<'_>,
    boost: f32,
    collector: &mut dyn ScoringCollector,
) -> Result<()> {
    let Some((info, points)) = points_of(leaf, field)? else {
        return Ok(());
    };
    if let Some(pf) = points.field(info.number) {
        check_shape_tree(&info.name, pf)?;
    }
    spatial_score_leaf(leaf, info, &points, visitor, rel, boost, collector)
}

// ---------------------------------------------------------------- component

/// `LatLonShapeQuery.getSpatialVisitor(component2D)` /
/// `XYShapeQuery.getSpatialVisitor(component2D)`: the cell relation and the
/// triangle predicates of a `Component2D`, in the encoding's decoded space.
struct ComponentVisitor {
    component: Arc<dyn Component2D>,
    encoding: ShapeEncoding,
}

impl std::fmt::Debug for ComponentVisitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComponentVisitor")
            .field("component", &self.component)
            .field("encoding", &self.encoding)
            .finish()
    }
}

impl SpatialVisitor for ComponentVisitor {
    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        let e = self.encoding;
        let min_y = e.decode_y(int_at(min, 0));
        let min_x = e.decode_x(int_at(min, ShapeField::BYTES));
        let max_y = e.decode_y(int_at(max, 2 * ShapeField::BYTES));
        let max_x = e.decode_x(int_at(max, 3 * ShapeField::BYTES));
        // check internal node against query
        self.component.relate(min_x, max_x, min_y, max_y)
    }

    fn intersects(&self, packed: &[u8]) -> bool {
        let Some(t) = triangle(packed) else {
            return false;
        };
        let t = ShapeField::decode_triangle(t);
        let (e, c) = (self.encoding, &self.component);
        match t.kind {
            TriangleType::Point => c.contains(e.decode_x(t.a_x), e.decode_y(t.a_y)),
            TriangleType::Line => c.intersects_line(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
            ),
            TriangleType::Triangle => c.intersects_triangle(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
                e.decode_x(t.c_x),
                e.decode_y(t.c_y),
            ),
        }
    }

    fn within(&self, packed: &[u8]) -> bool {
        let Some(t) = triangle(packed) else {
            return false;
        };
        let t = ShapeField::decode_triangle(t);
        let (e, c) = (self.encoding, &self.component);
        match t.kind {
            TriangleType::Point => c.contains(e.decode_x(t.a_x), e.decode_y(t.a_y)),
            TriangleType::Line => c.contains_line(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
            ),
            TriangleType::Triangle => c.contains_triangle(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
                e.decode_x(t.c_x),
                e.decode_y(t.c_y),
            ),
        }
    }

    fn contains(&self, packed: &[u8]) -> std::result::Result<WithinRelation, GeoError> {
        let Some(t) = triangle(packed) else {
            return Ok(WithinRelation::Disjoint);
        };
        let t = ShapeField::decode_triangle(t);
        let (e, c) = (self.encoding, &self.component);
        match t.kind {
            TriangleType::Point => c.within_point(e.decode_x(t.a_x), e.decode_y(t.a_y)),
            TriangleType::Line => c.within_line(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                t.ab,
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
            ),
            TriangleType::Triangle => c.within_triangle(
                e.decode_x(t.a_x),
                e.decode_y(t.a_y),
                t.ab,
                e.decode_x(t.b_x),
                e.decode_y(t.b_y),
                t.bc,
                e.decode_x(t.c_x),
                e.decode_y(t.c_y),
                t.ca,
            ),
        }
    }
}

/// `LatLonShapeQuery` (`LatLonShape.newGeometryQuery` and its line, polygon,
/// point and distance forms): the documents whose shape relates to the
/// geometries as `query_relation` says, at a constant score.
#[derive(Debug, Clone)]
pub struct LatLonShapeQuery {
    pub field: String,
    pub query_relation: QueryRelation,
    pub geometries: Vec<LatLonGeometry>,
    visitor: Arc<ComponentVisitor>,
}

impl LatLonShapeQuery {
    /// `LatLonShapeQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// A line under `WITHIN`, or geometries `LatLonGeometry.create` rejects,
    /// with Java's message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Self> {
        // validateGeometries
        if query_relation == QueryRelation::Within
            && geometries
                .iter()
                .any(|g| matches!(g, LatLonGeometry::Line(_)))
        {
            return Err(illegal(
                "LatLonShapeQuery does not support WITHIN queries with line geometries",
            ));
        }
        let component: Arc<dyn Component2D> =
            Arc::from(LatLonGeometry::create(geometries).map_err(geo)?);
        Ok(LatLonShapeQuery {
            field: field.into(),
            query_relation,
            geometries: geometries.to_vec(),
            visitor: Arc::new(ComponentVisitor {
                component,
                encoding: ShapeEncoding::LatLon,
            }),
        })
    }
}

impl DocumentQuery for LatLonShapeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        score_shape_leaf(
            &self.field,
            self.visitor.as_ref(),
            self.query_relation,
            leaf,
            boost,
            collector,
        )
    }
}

/// `XYShapeQuery` (every `XYShape` query): the documents whose cartesian
/// shape relates to the geometries as `query_relation` says.
#[derive(Debug, Clone)]
pub struct XYShapeQuery {
    pub field: String,
    pub query_relation: QueryRelation,
    pub geometries: Vec<XYGeometry>,
    visitor: Arc<ComponentVisitor>,
}

impl XYShapeQuery {
    /// `XYShapeQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// Geometries `XYGeometry.create` rejects, with Java's message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[XYGeometry],
    ) -> Result<Self> {
        let component: Arc<dyn Component2D> =
            Arc::from(XYGeometry::create(geometries).map_err(geo)?);
        Ok(XYShapeQuery {
            field: field.into(),
            query_relation,
            geometries: geometries.to_vec(),
            visitor: Arc::new(ComponentVisitor {
                component,
                encoding: ShapeEncoding::XY,
            }),
        })
    }
}

impl DocumentQuery for XYShapeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        score_shape_leaf(
            &self.field,
            self.visitor.as_ref(),
            self.query_relation,
            leaf,
            boost,
            collector,
        )
    }
}

// ---------------------------------------------------------------- bounding box

/// `Math.max`/`min` on `int`s widened, as `GeoUtils` takes `double`s.
#[inline]
fn d(v: i32) -> f64 {
    f64::from(v)
}

/// `SpatialQuery.EncodedRectangle`: a box in the encoded space, possibly
/// wrapping the coordinate system (crossing the dateline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EncodedRectangle {
    pub min_x: i32,
    pub max_x: i32,
    pub min_y: i32,
    pub max_y: i32,
    pub wraps_coordinate_system: bool,
}

impl EncodedRectangle {
    /// `contains(x, y)`.
    pub fn contains(&self, x: i32, y: i32) -> bool {
        if y < self.min_y || y > self.max_y {
            return false;
        }
        if self.wraps_coordinate_system {
            !(x > self.max_x && x < self.min_x)
        } else {
            !(x > self.max_x || x < self.min_x)
        }
    }

    /// `intersectsLine(aX, aY, bX, bY)`.
    pub fn intersects_line(&self, a_x: i32, a_y: i32, b_x: i32, b_y: i32) -> bool {
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) {
            return true;
        }
        // check bounding boxes are disjoint
        if a_y.max(b_y) < self.min_y || a_y.min(b_y) > self.max_y {
            return false;
        }
        if self.wraps_coordinate_system {
            // crosses dateline
            if a_x.min(b_x) > self.max_x && a_x.max(b_x) < self.min_x {
                return false;
            }
        } else if a_x.min(b_x) > self.max_x || a_x.max(b_x) < self.min_x {
            return false;
        }
        // expensive part
        self.edge_intersects_query(a_x, a_y, b_x, b_y)
    }

    /// `intersectsTriangle(aX, aY, bX, bY, cX, cY)`.
    pub fn intersects_triangle(
        &self,
        a_x: i32,
        a_y: i32,
        b_x: i32,
        b_y: i32,
        c_x: i32,
        c_y: i32,
    ) -> bool {
        // query contains any triangle points
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) || self.contains(c_x, c_y) {
            return true;
        }
        // check bounding box of triangle
        let t_min_y = a_y.min(b_y).min(c_y);
        let t_max_y = a_y.max(b_y).max(c_y);
        // check bounding boxes are disjoint
        if t_max_y < self.min_y || t_min_y > self.max_y {
            return false;
        }
        let t_min_x = a_x.min(b_x).min(c_x);
        let t_max_x = a_x.max(b_x).max(c_x);
        if self.wraps_coordinate_system {
            if t_min_x > self.max_x && t_max_x < self.min_x {
                return false;
            }
        } else if t_min_x > self.max_x || t_max_x < self.min_x {
            return false;
        }
        // expensive part
        point_in_triangle(
            d(t_min_x),
            d(t_max_x),
            d(t_min_y),
            d(t_max_y),
            d(self.min_x),
            d(self.min_y),
            d(a_x),
            d(a_y),
            d(b_x),
            d(b_y),
            d(c_x),
            d(c_y),
        ) || self.edge_intersects_query(a_x, a_y, b_x, b_y)
            || self.edge_intersects_query(b_x, b_y, c_x, c_y)
            || self.edge_intersects_query(c_x, c_y, a_x, a_y)
    }

    /// `intersectsRectangle(minX, maxX, minY, maxY)`.
    pub fn intersects_rectangle(&self, min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> bool {
        // simple Y check
        if self.min_y > max_y || self.max_y < min_y {
            return false;
        }
        if self.min_x <= max_x && (self.wraps_coordinate_system || self.max_x >= min_x) {
            return true;
        }
        self.wraps_coordinate_system
    }

    /// `containsRectangle(minX, maxX, minY, maxY)`.
    pub fn contains_rectangle(&self, min_x: i32, max_x: i32, min_y: i32, max_y: i32) -> bool {
        self.min_x <= min_x && self.max_x >= max_x && self.min_y <= min_y && self.max_y >= max_y
    }

    /// `containsLine(aX, aY, bX, bY)`.
    pub fn contains_line(&self, a_x: i32, a_y: i32, b_x: i32, b_y: i32) -> bool {
        if a_y < self.min_y || b_y < self.min_y || a_y > self.max_y || b_y > self.max_y {
            return false;
        }
        if self.wraps_coordinate_system {
            (a_x >= self.min_x && b_x >= self.min_x) || (a_x <= self.max_x && b_x <= self.max_x)
        } else {
            a_x >= self.min_x && b_x >= self.min_x && a_x <= self.max_x && b_x <= self.max_x
        }
    }

    /// `containsTriangle(aX, aY, bX, bY, cX, cY)`.
    pub fn contains_triangle(
        &self,
        a_x: i32,
        a_y: i32,
        b_x: i32,
        b_y: i32,
        c_x: i32,
        c_y: i32,
    ) -> bool {
        if a_y < self.min_y
            || b_y < self.min_y
            || c_y < self.min_y
            || a_y > self.max_y
            || b_y > self.max_y
            || c_y > self.max_y
        {
            return false;
        }
        if self.wraps_coordinate_system {
            (a_x >= self.min_x && b_x >= self.min_x && c_x >= self.min_x)
                || (a_x <= self.max_x && b_x <= self.max_x && c_x <= self.max_x)
        } else {
            a_x >= self.min_x
                && b_x >= self.min_x
                && c_x >= self.min_x
                && a_x <= self.max_x
                && b_x <= self.max_x
                && c_x <= self.max_x
        }
    }

    /// `withinLine(aX, aY, ab, bX, bY)`.
    pub fn within_line(&self, a_x: i32, a_y: i32, ab: bool, b_x: i32, b_y: i32) -> WithinRelation {
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) {
            return WithinRelation::NotWithin;
        }
        if ab
            && edge_intersects_box(
                a_x, a_y, b_x, b_y, self.min_x, self.max_x, self.min_y, self.max_y,
            )
        {
            return WithinRelation::NotWithin;
        }
        WithinRelation::Disjoint
    }

    /// `withinTriangle(aX, aY, ab, bX, bY, bc, cX, cY, ca)`.
    #[allow(clippy::too_many_arguments)]
    pub fn within_triangle(
        &self,
        a_x: i32,
        a_y: i32,
        ab: bool,
        b_x: i32,
        b_y: i32,
        bc: bool,
        c_x: i32,
        c_y: i32,
        ca: bool,
    ) -> WithinRelation {
        // Points belong to the shape so if points are inside the rectangle
        // then it cannot be within.
        if self.contains(a_x, a_y) || self.contains(b_x, b_y) || self.contains(c_x, c_y) {
            return WithinRelation::NotWithin;
        }
        // Bounding boxes disjoint?
        let t_min_y = a_y.min(b_y).min(c_y);
        let t_max_y = a_y.max(b_y).max(c_y);
        if t_max_y < self.min_y || t_min_y > self.max_y {
            return WithinRelation::Disjoint;
        }
        let t_min_x = a_x.min(b_x).min(c_x);
        let t_max_x = a_x.max(b_x).max(c_x);
        if self.wraps_coordinate_system {
            if t_min_x > self.max_x && t_max_x < self.min_x {
                return WithinRelation::Disjoint;
            }
        } else if t_min_x > self.max_x || t_max_x < self.min_x {
            return WithinRelation::Disjoint;
        }
        // If any of the edges intersects an edge belonging to the shape then
        // it cannot be within.
        let mut relation = WithinRelation::Disjoint;
        let (x0, x1, y0, y1) = (self.min_x, self.max_x, self.min_y, self.max_y);
        if edge_intersects_box(a_x, a_y, b_x, b_y, x0, x1, y0, y1) {
            if ab {
                return WithinRelation::NotWithin;
            }
            relation = WithinRelation::Candidate;
        }
        if edge_intersects_box(b_x, b_y, c_x, c_y, x0, x1, y0, y1) {
            if bc {
                return WithinRelation::NotWithin;
            }
            relation = WithinRelation::Candidate;
        }
        if edge_intersects_box(c_x, c_y, a_x, a_y, x0, x1, y0, y1) {
            if ca {
                return WithinRelation::NotWithin;
            }
            relation = WithinRelation::Candidate;
        }
        // Check if shape is within the triangle
        if relation == WithinRelation::Candidate
            || point_in_triangle(
                d(t_min_x),
                d(t_max_x),
                d(t_min_y),
                d(t_max_y),
                d(self.min_x),
                d(self.min_y),
                d(a_x),
                d(a_y),
                d(b_x),
                d(b_y),
                d(c_x),
                d(c_y),
            )
        {
            return WithinRelation::Candidate;
        }
        relation
    }

    /// `edgeIntersectsQuery(aX, aY, bX, bY)`.
    fn edge_intersects_query(&self, a_x: i32, a_y: i32, b_x: i32, b_y: i32) -> bool {
        if self.wraps_coordinate_system {
            return edge_intersects_box(
                a_x,
                a_y,
                b_x,
                b_y,
                GeoEncodingUtils::MIN_LON_ENCODED,
                self.max_x,
                self.min_y,
                self.max_y,
            ) || edge_intersects_box(
                a_x,
                a_y,
                b_x,
                b_y,
                self.min_x,
                GeoEncodingUtils::MAX_LON_ENCODED,
                self.min_y,
                self.max_y,
            );
        }
        edge_intersects_box(
            a_x, a_y, b_x, b_y, self.min_x, self.max_x, self.min_y, self.max_y,
        )
    }
}

/// `EncodedRectangle.edgeIntersectsBox`: the edge `a`-`b` touches the box's
/// boundary.
#[allow(clippy::too_many_arguments)]
fn edge_intersects_box(
    a_x: i32,
    a_y: i32,
    b_x: i32,
    b_y: i32,
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
) -> bool {
    if a_x.max(b_x) < min_x || a_x.min(b_x) > max_x || a_y.min(b_y) > max_y || a_y.max(b_y) < min_y
    {
        return false;
    }
    let (ax, ay, bx, by) = (d(a_x), d(a_y), d(b_x), d(b_y));
    let (x0, x1, y0, y1) = (d(min_x), d(max_x), d(min_y), d(max_y));
    GeoUtils::line_crosses_line_with_boundary(ax, ay, bx, by, x0, y1, x1, y1) // top
        || GeoUtils::line_crosses_line_with_boundary(ax, ay, bx, by, x1, y1, x1, y0) // bottom
        || GeoUtils::line_crosses_line_with_boundary(ax, ay, bx, by, x1, y0, x0, y0) // left
        || GeoUtils::line_crosses_line_with_boundary(ax, ay, bx, by, x0, y0, x0, y1)
    // right
}

/// One encoded box as the `int`s of `encode(minX, maxX, minY, maxY, b)`.
#[derive(Debug, Clone, Copy)]
struct BoxInts {
    min_x: i32,
    max_x: i32,
    min_y: i32,
    max_y: i32,
}

/// A cell's bounds: the four indexed dimensions (`minY, minX, maxY, maxX`)
/// of its min and max packed values.
#[derive(Debug, Clone, Copy)]
struct Cell {
    min: [i32; 4],
    max: [i32; 4],
}

impl Cell {
    fn new(min: &[u8], max: &[u8]) -> Cell {
        let four = |p: &[u8]| {
            [
                int_at(p, 0),
                int_at(p, ShapeField::BYTES),
                int_at(p, 2 * ShapeField::BYTES),
                int_at(p, 3 * ShapeField::BYTES),
            ]
        };
        Cell {
            min: four(min),
            max: four(max),
        }
    }
}

const MIN_Y: usize = 0;
const MIN_X: usize = 1;
const MAX_Y: usize = 2;
const MAX_X: usize = 3;

impl BoxInts {
    /// `disjoint(bbox, ...)`.
    fn disjoint(&self, c: &Cell) -> bool {
        c.min[MIN_X] > self.max_x
            || c.max[MAX_X] < self.min_x
            || c.min[MIN_Y] > self.max_y
            || c.max[MAX_Y] < self.min_y
    }

    /// `compareBBoxToRangeBBox(bbox, ...)`.
    fn compare(&self, c: &Cell) -> Relation {
        // check bounding box (DISJOINT)
        if self.disjoint(c) {
            return Relation::CellOutsideQuery;
        }
        if c.min[MIN_X] >= self.min_x
            && c.max[MAX_X] <= self.max_x
            && c.min[MIN_Y] >= self.min_y
            && c.max[MAX_Y] <= self.max_y
        {
            return Relation::CellInsideQuery;
        }
        Relation::CellCrossesQuery
    }

    /// `intersectBBoxWithRangeBBox(bbox, ...)`: a cell is inside for an
    /// intersection when every triangle in it has at least one corner of
    /// its bounding box in the query box.
    fn intersect(&self, c: &Cell) -> Relation {
        // check bounding box (DISJOINT)
        if self.disjoint(c) {
            return Relation::CellOutsideQuery;
        }
        if c.min[MIN_X] >= self.min_x && c.min[MIN_Y] >= self.min_y {
            if c.max[MIN_X] <= self.max_x && c.max[MAX_Y] <= self.max_y {
                return Relation::CellInsideQuery;
            }
            if c.max[MAX_X] <= self.max_x && c.max[MIN_Y] <= self.max_y {
                return Relation::CellInsideQuery;
            }
        }
        if c.max[MAX_X] <= self.max_x && c.max[MAX_Y] <= self.max_y {
            if c.min[MIN_X] >= self.min_x && c.min[MAX_Y] >= self.min_y {
                return Relation::CellInsideQuery;
            }
            if c.min[MAX_X] >= self.min_x && c.min[MIN_Y] >= self.min_y {
                return Relation::CellInsideQuery;
            }
        }
        Relation::CellCrossesQuery
    }
}

/// `LatLonShapeBoundingBoxQuery.EncodedLatLonRectangle`: the query box in
/// the encoded space, and the one or two boxes (`bbox`, and `west` across
/// the dateline) cells are compared with.
#[derive(Debug, Clone, Copy)]
struct EncodedLatLonRectangle {
    rect: EncodedRectangle,
    bbox: BoxInts,
    west: Option<BoxInts>,
}

impl EncodedLatLonRectangle {
    fn new(min_lat: f64, max_lat: f64, min_lon: f64, max_lon: f64) -> Result<Self> {
        let valid_min_lon = Self::validate_min_lon(min_lon, max_lon);
        let rect = EncodedRectangle {
            min_x: GeoEncodingUtils::encode_longitude_ceil(valid_min_lon).map_err(geo)?,
            max_x: GeoEncodingUtils::encode_longitude(max_lon).map_err(geo)?,
            min_y: GeoEncodingUtils::encode_latitude_ceil(min_lat).map_err(geo)?,
            max_y: GeoEncodingUtils::encode_latitude(max_lat).map_err(geo)?,
            wraps_coordinate_system: valid_min_lon > max_lon,
        };
        let (bbox, west) = if rect.wraps_coordinate_system {
            // crossing dateline is split into east/west boxes
            (
                BoxInts {
                    min_x: rect.min_x,
                    max_x: GeoEncodingUtils::MAX_LON_ENCODED,
                    min_y: rect.min_y,
                    max_y: rect.max_y,
                },
                Some(BoxInts {
                    min_x: GeoEncodingUtils::MIN_LON_ENCODED,
                    max_x: rect.max_x,
                    min_y: rect.min_y,
                    max_y: rect.max_y,
                }),
            )
        } else {
            (
                BoxInts {
                    min_x: rect.min_x,
                    max_x: rect.max_x,
                    min_y: rect.min_y,
                    max_y: rect.max_y,
                },
                None,
            )
        };
        Ok(EncodedLatLonRectangle { rect, bbox, west })
    }

    /// `validateMinLon(minLon, maxLon)`: -180 for a box from the dateline
    /// across it.
    #[allow(clippy::float_cmp)]
    fn validate_min_lon(min_lon: f64, max_lon: f64) -> f64 {
        if min_lon == 180.0 && min_lon > max_lon {
            return -180.0;
        }
        min_lon
    }

    /// `relateRangeBBox`.
    fn relate_range_bbox(&self, c: &Cell) -> Relation {
        let east = self.bbox.compare(c);
        match self.west {
            Some(w) if east == Relation::CellOutsideQuery => w.compare(c),
            _ => east,
        }
    }

    /// `intersectRangeBBox`.
    fn intersect_range_bbox(&self, c: &Cell) -> Relation {
        let east = self.bbox.intersect(c);
        match self.west {
            Some(w) if east == Relation::CellOutsideQuery => w.intersect(c),
            _ => east,
        }
    }
}

/// `LatLonShapeBoundingBoxQuery.getSpatialVisitor()`.
#[derive(Debug)]
struct BoxVisitor {
    rect: EncodedLatLonRectangle,
    query_relation: QueryRelation,
}

impl SpatialVisitor for BoxVisitor {
    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        let c = Cell::new(min, max);
        if matches!(
            self.query_relation,
            QueryRelation::Intersects | QueryRelation::Disjoint
        ) {
            return self.rect.intersect_range_bbox(&c);
        }
        self.rect.relate_range_bbox(&c)
    }

    fn intersects(&self, packed: &[u8]) -> bool {
        let Some(t) = triangle(packed) else {
            return false;
        };
        let t = ShapeField::decode_triangle(t);
        let r = &self.rect.rect;
        match t.kind {
            TriangleType::Point => r.contains(t.a_x, t.a_y),
            TriangleType::Line => r.intersects_line(t.a_x, t.a_y, t.b_x, t.b_y),
            TriangleType::Triangle => {
                r.intersects_triangle(t.a_x, t.a_y, t.b_x, t.b_y, t.c_x, t.c_y)
            }
        }
    }

    fn within(&self, packed: &[u8]) -> bool {
        let Some(t) = triangle(packed) else {
            return false;
        };
        let t = ShapeField::decode_triangle(t);
        let r = &self.rect.rect;
        match t.kind {
            TriangleType::Point => r.contains(t.a_x, t.a_y),
            TriangleType::Line => r.contains_line(t.a_x, t.a_y, t.b_x, t.b_y),
            TriangleType::Triangle => r.contains_triangle(t.a_x, t.a_y, t.b_x, t.b_y, t.c_x, t.c_y),
        }
    }

    fn contains(&self, packed: &[u8]) -> std::result::Result<WithinRelation, GeoError> {
        let Some(t) = triangle(packed) else {
            return Ok(WithinRelation::Disjoint);
        };
        // decode indexed triangle
        let t = ShapeField::decode_triangle(t);
        let r = &self.rect.rect;
        Ok(match t.kind {
            TriangleType::Point => {
                if r.contains(t.a_x, t.a_y) {
                    WithinRelation::NotWithin
                } else {
                    WithinRelation::Disjoint
                }
            }
            TriangleType::Line => r.within_line(t.a_x, t.a_y, t.ab, t.b_x, t.b_y),
            TriangleType::Triangle => {
                r.within_triangle(t.a_x, t.a_y, t.ab, t.b_x, t.b_y, t.bc, t.c_x, t.c_y, t.ca)
            }
        })
    }

    fn check_contains(&self) -> std::result::Result<(), GeoError> {
        if self.rect.rect.wraps_coordinate_system {
            return Err(GeoError::IllegalArgument(
                "withinTriangle is not supported for rectangles crossing the date line".into(),
            ));
        }
        Ok(())
    }
}

/// `LatLonShapeBoundingBoxQuery` (`LatLonShape.newBoxQuery`): a box query
/// related in the encoded space.
#[derive(Debug, Clone)]
pub struct LatLonShapeBoundingBoxQuery {
    pub field: String,
    pub query_relation: QueryRelation,
    pub rectangle: Rectangle,
    visitor: Arc<BoxVisitor>,
}

impl LatLonShapeBoundingBoxQuery {
    /// `LatLonShapeBoundingBoxQuery(field, queryRelation, rectangle)`.
    ///
    /// # Errors
    /// None for a valid [`Rectangle`]; the encoding's otherwise.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        rectangle: Rectangle,
    ) -> Result<Self> {
        // createComponent2D is not used by the query (Java builds it anyway).
        let rect = EncodedLatLonRectangle::new(
            rectangle.min_lat,
            rectangle.max_lat,
            rectangle.min_lon,
            rectangle.max_lon,
        )?;
        Ok(LatLonShapeBoundingBoxQuery {
            field: field.into(),
            query_relation,
            rectangle,
            visitor: Arc::new(BoxVisitor {
                rect,
                query_relation,
            }),
        })
    }
}

impl DocumentQuery for LatLonShapeBoundingBoxQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        score_shape_leaf(
            &self.field,
            self.visitor.as_ref(),
            self.query_relation,
            leaf,
            boost,
            collector,
        )
    }
}

// ---------------------------------------------------------------- doc values

/// `BaseShapeDocValuesQuery`'s shared state: the field, the relation and
/// the query geometries as one `Component2D`.
#[derive(Debug, Clone)]
struct ShapeDocValuesMatcher {
    field: String,
    query_relation: QueryRelation,
    component: Arc<dyn Component2D>,
    encoding: ShapeEncoding,
}

impl ShapeDocValuesMatcher {
    /// `validateRelation(queryRelation)`.
    fn validate_relation(query_relation: QueryRelation) -> Result<()> {
        if query_relation == QueryRelation::Contains {
            return Err(illegal(
                "ShapeDocValuesBoundingBoxQuery does not yet support CONTAINS queries",
            ));
        }
        Ok(())
    }

    /// `matchesComponent(dv, queryRelation, component)`.
    fn matches_component(&self, dv: &ShapeDocValues<'_>) -> Result<bool> {
        let r = dv
            .relate(self.component.as_ref())
            .map_err(|e| illegal(e.to_string()))?;
        if r != Relation::CellOutsideQuery {
            if self.query_relation == QueryRelation::Within {
                return Ok(r == Relation::CellInsideQuery);
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// `match(shapeDocValues)`: DISJOINT is the negation.
    fn matches(&self, dv: &ShapeDocValues<'_>) -> Result<bool> {
        let result = self.matches_component(dv)?;
        if self.query_relation == QueryRelation::Disjoint {
            return Ok(!result);
        }
        Ok(result)
    }

    /// `getScorerSupplier`: every document with a value that matches.
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        // `getBinaryDocValues(field)`: `null` unless the field is BINARY.
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        if info.doc_values_type != DocValuesType::Binary {
            return Ok(());
        }
        let r = reader(leaf)?;
        let Some((meta, data)) = r.doc_values_for_field(info.number) else {
            return Ok(());
        };
        let Some(entry) = meta.binary_entry(info.number) else {
            return Ok(());
        };
        let mut values = BinaryReader::new(data, entry);
        for doc in 0..r.max_doc {
            let Some(bytes) = values.value(doc)? else {
                continue;
            };
            // getShapeDocValues(values.binaryValue()). Java's constructor
            // also builds the centroid `Point` and bounding `Rectangle`,
            // which for lat/lon cannot fail (every decoded value is in
            // range, and `Rectangle` only asserts its order), so that work
            // is skipped there; an `XYRectangle` can refuse a corrupt box,
            // so the cartesian value is opened in full.
            let dv = match self.encoding {
                ShapeEncoding::LatLon => {
                    let dv =
                        ShapeDocValues::from_bytes(ShapeEncoding::LatLon, Cow::Borrowed(bytes))
                            .map_err(|e| illegal(e.to_string()))?;
                    self.matches(&dv)?
                }
                ShapeEncoding::XY => {
                    let dv = XYShapeDocValues::new(Cow::Borrowed(bytes))
                        .map_err(|e| illegal(e.to_string()))?;
                    self.matches(dv.values())?
                }
            };
            if dv {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `LatLonShapeDocValuesQuery` (`LatLonShape.newSlowDocValuesBoxQuery`;
/// Java's constructor, package-private there, takes any geometries): the
/// documents whose shape doc value relates to the geometries as
/// `query_relation` says. `CONTAINS` is refused.
#[derive(Debug, Clone)]
pub struct LatLonShapeDocValuesQuery {
    pub geometries: Vec<LatLonGeometry>,
    m: ShapeDocValuesMatcher,
}

impl LatLonShapeDocValuesQuery {
    /// `LatLonShapeDocValuesQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// `CONTAINS`, or geometries `LatLonGeometry.create` rejects, with
    /// Java's message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Self> {
        ShapeDocValuesMatcher::validate_relation(query_relation)?;
        let component: Arc<dyn Component2D> =
            Arc::from(LatLonGeometry::create(geometries).map_err(geo)?);
        Ok(LatLonShapeDocValuesQuery {
            geometries: geometries.to_vec(),
            m: ShapeDocValuesMatcher {
                field: field.into(),
                query_relation,
                component,
                encoding: ShapeEncoding::LatLon,
            },
        })
    }

    /// The field.
    pub fn field(&self) -> &str {
        &self.m.field
    }

    /// The relation.
    pub fn query_relation(&self) -> QueryRelation {
        self.m.query_relation
    }
}

impl DocumentQuery for LatLonShapeDocValuesQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        self.m.score_leaf(leaf, boost, collector)
    }
}

/// `XYShapeDocValuesQuery` (`XYShape.newSlowDocValuesBoxQuery`; any
/// geometries through the constructor): the cartesian form of
/// [`LatLonShapeDocValuesQuery`].
#[derive(Debug, Clone)]
pub struct XYShapeDocValuesQuery {
    pub geometries: Vec<XYGeometry>,
    m: ShapeDocValuesMatcher,
}

impl XYShapeDocValuesQuery {
    /// `XYShapeDocValuesQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// `CONTAINS`, or geometries `XYGeometry.create` rejects, with Java's
    /// message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[XYGeometry],
    ) -> Result<Self> {
        ShapeDocValuesMatcher::validate_relation(query_relation)?;
        let component: Arc<dyn Component2D> =
            Arc::from(XYGeometry::create(geometries).map_err(geo)?);
        Ok(XYShapeDocValuesQuery {
            geometries: geometries.to_vec(),
            m: ShapeDocValuesMatcher {
                field: field.into(),
                query_relation,
                component,
                encoding: ShapeEncoding::XY,
            },
        })
    }

    /// The field.
    pub fn field(&self) -> &str {
        &self.m.field
    }

    /// The relation.
    pub fn query_relation(&self) -> QueryRelation {
        self.m.query_relation
    }
}

impl DocumentQuery for XYShapeDocValuesQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        self.m.score_leaf(leaf, boost, collector)
    }
}

#[cfg(test)]
mod tests;
