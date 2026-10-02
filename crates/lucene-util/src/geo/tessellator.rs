//! Port of `org.apache.lucene.geo.Tessellator`: ear-clipping triangulation
//! (mapbox's earcut, after FIST and Eberly) of a polygon with holes into the
//! triangles `LatLonShape`/`XYShape` index.
//!
//! The triangles are part of the index format -- each becomes one point of
//! the shape field's BKD tree -- so the port produces Java's triangles in
//! Java's order, or fails with Java's message where Java fails. That holds
//! only if every heuristic here (ear order, hole bridging, the morton-order
//! fast paths, the cure/split fallbacks) is Java's, so the algorithm is
//! transcribed function for function.
//!
//! Rust-only differences:
//!
//! - Java's `Node` objects linked by references become an arena (`Vec<Node>`)
//!   linked by indices; `new Node(other)` appends a copy. A node keeps its
//!   coordinate values instead of a reference to the polygon's arrays (Java
//!   never mutates those arrays, so this reads the same values).
//! - The hole lookup `Map<Node, Polygon>` (identity-keyed) is keyed by arena
//!   index.
//! - The unused `Object polygon` argument threaded through Java's
//!   `earcutLinkedList`/`splitEarcut` is dropped.
//! - Exceptions become [`GeoError::IllegalArgument`] with Java's messages.
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;

use super::geo_encoding_utils::GeoEncodingUtils;
use super::geo_utils::{GeoUtils, WindingOrder};
use super::point::Point;
use super::polygon::Polygon;
use super::xy_encoding_utils::XYEncodingUtils;
use super::xy_polygon::XYPolygon;
use super::{java_double_string, java_min, GeoError};
use crate::bit_util;
use crate::strict_math;

/// Dumb heuristic: above this many vertices, use sorted morton values.
const VERTEX_THRESHOLD: i32 = 80;

/// `Monitor.FAILED`.
pub const FAILED: &str = "FAILED";
/// `Monitor.COMPLETED`.
pub const COMPLETED: &str = "COMPLETED";

/// State of the tessellated split (avoids recursion).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Init,
    Cure,
    Split,
}

impl State {
    fn name(self) -> &'static str {
        match self {
            State::Init => "INIT",
            State::Cure => "CURE",
            State::Split => "SPLIT",
        }
    }
}

/// `Tessellator.Monitor`: receives the internal state at each step of the
/// triangulation, for debugging.
pub trait Monitor {
    /// `currentState(status, points, tessellation)`: called on each loop of
    /// the main ear-clipping algorithm. `points` is `None` where Java passes
    /// `null`.
    fn current_state(&mut self, status: &str, points: Option<&[Point]>, tessellation: &[Triangle]);
    /// `startSplit(status, leftPolygon, rightPolygon)`.
    fn start_split(&mut self, status: &str, left_polygon: &[Point], right_polygon: &[Point]);
    /// `endSplit(status)`.
    fn end_split(&mut self, status: &str);
}

/// `Tessellator.Triangle`: one triangle of the mesh. Vertices are indexed
/// 0..3; edge `i` runs from vertex `i` to vertex `(i + 1) % 3`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Triangle {
    encoded_x: [i32; 3],
    encoded_y: [i32; 3],
    x: [f64; 3],
    y: [f64; 3],
    edge_from_polygon: [bool; 3],
}

impl Triangle {
    /// `getEncodedX(vertex)`: the quantized x (encoded longitude).
    pub fn encoded_x(&self, vertex: usize) -> i32 {
        self.encoded_x[vertex]
    }

    /// `getEncodedY(vertex)`: the quantized y (encoded latitude).
    pub fn encoded_y(&self, vertex: usize) -> i32 {
        self.encoded_y[vertex]
    }

    /// `getX(vertex)`.
    pub fn x(&self, vertex: usize) -> f64 {
        self.x[vertex]
    }

    /// `getY(vertex)`.
    pub fn y(&self, vertex: usize) -> f64 {
        self.y[vertex]
    }

    /// `isEdgefromPolygon(startVertex)`: whether the edge is part of the
    /// original polygon's boundary.
    pub fn is_edge_from_polygon(&self, start_vertex: usize) -> bool {
        self.edge_from_polygon[start_vertex]
    }
}

impl std::fmt::Display for Triangle {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}, {} [{}] {}, {} [{}] {}, {} [{}]",
            self.encoded_x[0],
            self.encoded_y[0],
            self.edge_from_polygon[0],
            self.encoded_x[1],
            self.encoded_y[1],
            self.edge_from_polygon[1],
            self.encoded_x[2],
            self.encoded_y[2],
            self.edge_from_polygon[2]
        )
    }
}

/// `Tessellator.Node`: one vertex of the circular doubly-linked list.
#[derive(Debug, Clone, Copy)]
struct Node {
    /// node index in the linked list
    idx: i32,
    /// `getX()`: the polygon's coordinate.
    vx: f64,
    /// `getY()`.
    vy: f64,
    /// encoded x value
    x: i32,
    /// encoded y value
    y: i32,
    /// morton code for sorting
    morton: i64,
    previous: usize,
    next: usize,
    previous_z: Option<usize>,
    next_z: Option<usize>,
    /// if the edge from this node to the next node is part of the polygon edges
    is_next_edge_from_polygon: bool,
}

/// What `eliminateHoles` needs to know about each hole.
struct HoleInfo<'h> {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    /// `verticesToGeoJSON` of the hole, built only for the error message
    /// (as Java builds it only when it throws).
    geojson: Box<dyn Fn() -> String + 'h>,
}

/// The arena plus the optional monitor.
struct Tess<'m> {
    nodes: Vec<Node>,
    monitor: Option<&'m mut dyn Monitor>,
}

/// `Tessellator.tessellate(Polygon, checkSelfIntersections)`.
pub fn tessellate(
    polygon: &Polygon,
    check_self_intersections: bool,
) -> Result<Vec<Triangle>, GeoError> {
    tessellate_with_monitor(polygon, check_self_intersections, None)
}

/// `Tessellator.tessellate(Polygon, checkSelfIntersections, monitor)`.
pub fn tessellate_with_monitor(
    polygon: &Polygon,
    check_self_intersections: bool,
    monitor: Option<&mut dyn Monitor>,
) -> Result<Vec<Triangle>, GeoError> {
    let mut t = Tess {
        nodes: Vec::with_capacity(polygon.num_points() * 2),
        monitor,
    };
    let outer = t.create_doubly_linked_list(
        polygon.poly_lons(),
        polygon.poly_lats(),
        polygon.winding_order(),
        true,
        0,
        WindingOrder::CW,
    )?;
    let hole_points: Vec<usize> = polygon.holes().iter().map(Polygon::num_points).collect();
    let outer = t.check_outer(outer)?;
    let outer = if polygon.num_holes() > 0 {
        t.eliminate_holes_latlon(polygon, outer)?
    } else {
        outer
    };
    t.finish(
        outer,
        polygon.num_points(),
        &hole_points,
        check_self_intersections,
    )
}

/// `Tessellator.tessellate(XYPolygon, checkSelfIntersections)`.
pub fn tessellate_xy(
    polygon: &XYPolygon,
    check_self_intersections: bool,
) -> Result<Vec<Triangle>, GeoError> {
    tessellate_xy_with_monitor(polygon, check_self_intersections, None)
}

/// `Tessellator.tessellate(XYPolygon, checkSelfIntersections, monitor)`.
pub fn tessellate_xy_with_monitor(
    polygon: &XYPolygon,
    check_self_intersections: bool,
    monitor: Option<&mut dyn Monitor>,
) -> Result<Vec<Triangle>, GeoError> {
    let mut t = Tess {
        nodes: Vec::with_capacity(polygon.num_points() * 2),
        monitor,
    };
    let outer = t.create_doubly_linked_list(
        &XYEncodingUtils::float_array_to_double_array(polygon.poly_x()),
        &XYEncodingUtils::float_array_to_double_array(polygon.poly_y()),
        polygon.winding_order(),
        false,
        0,
        WindingOrder::CW,
    )?;
    let hole_points: Vec<usize> = polygon.holes().iter().map(XYPolygon::num_points).collect();
    let outer = t.check_outer(outer)?;
    let outer = if polygon.num_holes() > 0 {
        t.eliminate_holes_xy(polygon, outer)?
    } else {
        outer
    };
    t.finish(
        outer,
        polygon.num_points(),
        &hole_points,
        check_self_intersections,
    )
}

/// `Tessellator.linesIntersect`: whether two segments intersect.
pub fn lines_intersect(
    a_x0: f64,
    a_y0: f64,
    a_x1: f64,
    a_y1: f64,
    b_x0: f64,
    b_y0: f64,
    b_x1: f64,
    b_y1: f64,
) -> bool {
    (area(a_x0, a_y0, a_x1, a_y1, b_x0, b_y0) > 0.0)
        != (area(a_x0, a_y0, a_x1, a_y1, b_x1, b_y1) > 0.0)
        && (area(b_x0, b_y0, b_x1, b_y1, a_x0, a_y0) > 0.0)
            != (area(b_x0, b_y0, b_x1, b_y1, a_x1, a_y1) > 0.0)
}

/// Compute signed area of triangle, negative means convex angle and
/// positive reflex angle.
#[inline]
fn area(a_x: f64, a_y: f64, b_x: f64, b_y: f64, c_x: f64, c_y: f64) -> f64 {
    (b_y - a_y) * (c_x - b_x) - (b_x - a_x) * (c_y - b_y)
}

/// Compute whether point is in a candidate ear.
#[inline]
fn point_in_ear(x: f64, y: f64, ax: f64, ay: f64, bx: f64, by: f64, cx: f64, cy: f64) -> bool {
    (cx - x) * (ay - y) - (ax - x) * (cy - y) >= 0.0
        && (ax - x) * (by - y) - (bx - x) * (ay - y) >= 0.0
        && (bx - x) * (cy - y) - (cx - x) * (by - y) >= 0.0
}

/// `Long.compareUnsigned(a, b)` as an ordering test.
#[inline]
fn ucmp(a: i64) -> u64 {
    a as u64
}

#[inline]
fn flip(v: i32) -> i32 {
    v ^ i32::MIN
}

fn illegal_tessellate() -> GeoError {
    GeoError::illegal("Unable to Tessellate shape. Possible malformed shape detected.")
}

impl Tess<'_> {
    // ------------------------------------------------------------ node access

    #[inline]
    fn n(&self, i: usize) -> &Node {
        &self.nodes[i]
    }
    #[inline]
    fn nx(&self, i: usize) -> f64 {
        self.nodes[i].vx
    }
    #[inline]
    fn ny(&self, i: usize) -> f64 {
        self.nodes[i].vy
    }
    #[inline]
    fn next(&self, i: usize) -> usize {
        self.nodes[i].next
    }
    #[inline]
    fn prev(&self, i: usize) -> usize {
        self.nodes[i].previous
    }

    // ------------------------------------------------------------ entry

    fn check_outer(&self, outer: Option<usize>) -> Result<usize, GeoError> {
        // If an outer node hasn't been detected, the shape is malformed.
        let outer =
            outer.ok_or_else(|| GeoError::illegal("Malformed shape detected in Tessellator!"))?;
        if outer == self.next(outer) || outer == self.next(self.next(outer)) {
            return Err(GeoError::illegal(
                "at least three non-collinear points required",
            ));
        }
        Ok(outer)
    }

    fn finish(
        &mut self,
        outer: usize,
        num_points: usize,
        hole_points: &[usize],
        check_self_intersections: bool,
    ) -> Result<Vec<Triangle>, GeoError> {
        // If the shape crosses VERTEX_THRESHOLD, use z-order curve hashing:
        let mut threshold = VERTEX_THRESHOLD as i64 - num_points as i64;
        let mut i = 0;
        while threshold >= 0 && i < hole_points.len() {
            threshold -= hole_points[i] as i64;
            i += 1;
        }
        let morton_optimized = threshold < 0;
        if morton_optimized {
            self.sort_by_morton(outer);
        }
        if check_self_intersections {
            self.check_intersection(outer, morton_optimized)?;
        }
        let mut result = Vec::new();
        self.earcut_linked_list(Some(outer), &mut result, State::Init, morton_optimized, 0)?;
        if result.is_empty() {
            self.notify_monitor_status(FAILED, None, &result)?;
            return Err(illegal_tessellate());
        }
        self.notify_monitor_status(COMPLETED, None, &result)?;
        Ok(result)
    }

    /// `createDoublyLinkedList`: links the points in the requested winding
    /// order, drops a closing duplicate, filters, returns the last node.
    fn create_doubly_linked_list(
        &mut self,
        x: &[f64],
        y: &[f64],
        poly_winding_order: WindingOrder,
        is_geo: bool,
        mut start_index: i32,
        winding_order: WindingOrder,
    ) -> Result<Option<usize>, GeoError> {
        let mut last_node: Option<usize> = None;
        if winding_order == poly_winding_order {
            for i in 0..x.len() {
                last_node = Some(self.insert_node(x, y, start_index, i, last_node, is_geo)?);
                start_index += 1;
            }
        } else {
            for i in (0..x.len()).rev() {
                last_node = Some(self.insert_node(x, y, start_index, i, last_node, is_geo)?);
                start_index += 1;
            }
        }
        // if first and last node are the same then remove the end node and set lastNode to the start
        if let Some(last) = last_node {
            if self.is_vertex_equals(last, self.next(last)) {
                self.remove_node(last, true);
                last_node = Some(self.next(last));
            }
        }
        Ok(last_node.map(|l| self.filter_points(l, None)))
    }

    fn eliminate_holes_xy(&mut self, polygon: &XYPolygon, outer: usize) -> Result<usize, GeoError> {
        let mut hole_list = Vec::new();
        let mut infos = HashMap::new();
        let mut node_index = polygon.num_points() as i32;
        for hole in polygon.holes() {
            let list = self.create_doubly_linked_list(
                &XYEncodingUtils::float_array_to_double_array(hole.poly_x()),
                &XYEncodingUtils::float_array_to_double_array(hole.poly_y()),
                hole.winding_order(),
                false,
                node_index,
                WindingOrder::CCW,
            )?;
            if let Some(list) = list {
                let left_most = self.fetch_leftmost(list);
                hole_list.push(left_most);
                infos.insert(
                    left_most,
                    HoleInfo {
                        min_x: f64::from(hole.min_x),
                        max_x: f64::from(hole.max_x),
                        min_y: f64::from(hole.min_y),
                        max_y: f64::from(hole.max_y),
                        geojson: Box::new(move || {
                            XYPolygon::vertices_to_geojson(hole.poly_x(), hole.poly_y())
                        }),
                    },
                );
            }
            node_index += hole.num_points() as i32;
        }
        self.eliminate_holes(hole_list, &infos, outer)
    }

    fn eliminate_holes_latlon(
        &mut self,
        polygon: &Polygon,
        outer: usize,
    ) -> Result<usize, GeoError> {
        let mut hole_list = Vec::new();
        let mut infos = HashMap::new();
        let mut node_index = polygon.num_points() as i32;
        for hole in polygon.holes() {
            let list = self
                .create_doubly_linked_list(
                    hole.poly_lons(),
                    hole.poly_lats(),
                    hole.winding_order(),
                    true,
                    node_index,
                    WindingOrder::CCW,
                )?
                .expect("a hole has points");
            if list == self.next(list) {
                return Err(GeoError::illegal(format!(
                    "Points are all coplanar in hole: {hole}"
                )));
            }
            let left_most = self.fetch_leftmost(list);
            hole_list.push(left_most);
            infos.insert(
                left_most,
                HoleInfo {
                    min_x: hole.min_lon,
                    max_x: hole.max_lon,
                    min_y: hole.min_lat,
                    max_y: hole.max_lat,
                    geojson: Box::new(move || {
                        Polygon::vertices_to_geojson(hole.poly_lats(), hole.poly_lons())
                    }),
                },
            );
            node_index += hole.num_points() as i32;
        }
        self.eliminate_holes(hole_list, &infos, outer)
    }

    fn eliminate_holes(
        &mut self,
        mut hole_list: Vec<usize>,
        infos: &HashMap<usize, HoleInfo<'_>>,
        mut outer: usize,
    ) -> Result<usize, GeoError> {
        // Sort the hole vertices by x coordinate (List.sort: a stable sort).
        hole_list.sort_by(|&a, &b| {
            let mut diff = self.nx(a) - self.nx(b);
            if diff == 0.0 {
                diff = self.ny(a) - self.ny(b);
                if diff == 0.0 {
                    // same hole node
                    let ma = java_min(self.ny(self.prev(a)), self.ny(self.next(a)));
                    let mb = java_min(self.ny(self.prev(b)), self.ny(self.next(b)));
                    diff = ma - mb;
                }
            }
            if diff < 0.0 {
                std::cmp::Ordering::Less
            } else if diff > 0.0 {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        // Process holes from left to right.
        for hole_node in hole_list {
            let h = &infos[&hole_node];
            match self.eliminate_hole(hole_node, outer, h.min_x, h.max_x, h.min_y, h.max_y) {
                Some(result) => {
                    outer = self.filter_points(result, Some(self.next(result)));
                }
                None => {
                    let mut polygon = (h.geojson)();
                    if polygon.chars().count() > 100 {
                        polygon = format!("{}...", polygon.chars().take(100).collect::<String>());
                    }
                    return Err(GeoError::illegal(format!(
                        "Illegal hole detected: {polygon}"
                    )));
                }
            }
        }
        // Filter co-planar nodes and return a pointer to the list.
        Ok(self.filter_points(outer, None))
    }

    /// `eliminateHole`: bridges a hole to the outer ring; returns the node
    /// to use as the new outer node, or `None` if no bridge was found.
    fn eliminate_hole(
        &mut self,
        hole_node: usize,
        outer: usize,
        hole_min_x: f64,
        hole_max_x: f64,
        hole_min_y: f64,
        hole_max_y: f64,
    ) -> Option<usize> {
        // Attempt to merge the hole using a common point between if it exists.
        if let Some(merged) = self.maybe_merge_hole_with_shared_vertices(
            hole_node, outer, hole_min_x, hole_max_x, hole_min_y, hole_max_y,
        ) {
            return Some(merged);
        }
        // Attempt to find a logical bridge between the HoleNode and OuterNode.
        let bridge = self.fetch_hole_bridge(hole_node, outer)?;
        // compute if the bridge overlaps with a polygon edge.
        let from_polygon = self.is_point_in_line(bridge, self.next(bridge), hole_node)
            || self.is_point_in_line(bridge, self.prev(bridge), hole_node)
            || self.is_point_in_line(hole_node, self.next(hole_node), bridge)
            || self.is_point_in_line(hole_node, self.prev(hole_node), bridge);
        self.split_polygon(bridge, hole_node, from_polygon);
        Some(outer)
    }

    /// `maybeMergeHoleWithSharedVertices`.
    fn maybe_merge_hole_with_shared_vertices(
        &mut self,
        hole_node: usize,
        outer: usize,
        hole_min_x: f64,
        hole_max_x: f64,
        hole_min_y: f64,
        hole_max_y: f64,
    ) -> Option<usize> {
        let mut shared_vertex: Option<usize> = None;
        let mut shared_vertex_connection: Option<usize> = None;
        let mut leftmost_shared_vertex_connection: Option<usize> = None;
        let mut next = outer;
        loop {
            if super::rectangle::Rectangle::contains_point(
                self.ny(next),
                self.nx(next),
                hole_min_y,
                hole_max_y,
                hole_min_x,
                hole_max_x,
            ) {
                if let Some(new_shared_vertex) = self.get_shared_vertex(hole_node, next) {
                    // Check if this shared vertex is the leftmost point of the hole (holeNode)
                    if self.is_vertex_equals(new_shared_vertex, hole_node)
                        && leftmost_shared_vertex_connection.is_none()
                    {
                        leftmost_shared_vertex_connection = Some(next);
                    }
                    match shared_vertex {
                        None => {
                            shared_vertex = Some(new_shared_vertex);
                            shared_vertex_connection = Some(next);
                        }
                        Some(sv) if sv == new_shared_vertex => {
                            // Same vertex found again via different connection.
                            shared_vertex_connection = Some(self.get_shared_inside_vertex(
                                sv,
                                shared_vertex_connection.expect("set with shared_vertex"),
                                next,
                            ));
                        }
                        Some(_) => {}
                    }
                }
            }
            next = self.next(next);
            if next == outer {
                break;
            }
        }
        if let Some(lsvc) = leftmost_shared_vertex_connection {
            let svc = shared_vertex_connection.expect("leftmost implies shared");
            if self.n(lsvc).idx >= self.n(svc).idx {
                self.split_polygon(lsvc, hole_node, true);
                if lsvc != svc {
                    return Some(lsvc);
                }
                return Some(outer);
            }
        }
        if let Some(sv) = shared_vertex {
            self.split_polygon(
                shared_vertex_connection.expect("set with shared_vertex"),
                sv,
                true,
            );
            return Some(outer);
        }
        None
    }

    /// `getSharedVertex`: the node of `polygon`'s ring equal to `vertex`.
    fn get_shared_vertex(&self, polygon: usize, vertex: usize) -> Option<usize> {
        let mut next = polygon;
        loop {
            if self.is_vertex_equals(next, vertex) {
                return Some(next);
            }
            next = self.next(next);
            if next == polygon {
                return None;
            }
        }
    }

    /// `getSharedInsideVertex`: the candidate with the smaller angle to the
    /// hole vertex.
    fn get_shared_inside_vertex(
        &self,
        hole_vertex: usize,
        candidate_a: usize,
        candidate_b: usize,
    ) -> usize {
        let hv_next = self.next(hole_vertex);
        let a1 = area(
            self.nx(self.prev(candidate_a)),
            self.ny(self.prev(candidate_a)),
            self.nx(hole_vertex),
            self.ny(hole_vertex),
            self.nx(hv_next),
            self.ny(hv_next),
        );
        let a2 = area(
            self.nx(self.prev(candidate_b)),
            self.ny(self.prev(candidate_b)),
            self.nx(hole_vertex),
            self.ny(hole_vertex),
            self.nx(hv_next),
            self.ny(hv_next),
        );
        if (a1 < 0.0) != (a2 < 0.0) {
            // one is convex, the other reflex, get the convex one
            if a1 < a2 {
                candidate_a
            } else {
                candidate_b
            }
        } else {
            // both are convex / reflex, choose the smallest angle
            let angle1 = self.angle(self.prev(candidate_a), candidate_a, hv_next);
            let angle2 = self.angle(self.prev(candidate_b), candidate_b, hv_next);
            if angle1 < angle2 {
                candidate_a
            } else {
                candidate_b
            }
        }
    }

    fn angle(&self, a: usize, b: usize, c: usize) -> f64 {
        let ax = self.nx(a) - self.nx(b);
        let ay = self.ny(a) - self.ny(b);
        let cx = self.nx(c) - self.nx(b);
        let cy = self.ny(c) - self.ny(b);
        let dot_product = ax * cx + ay * cy;
        let a_length = (ax * ax + ay * ay).sqrt();
        let b_length = (cx * cx + cy * cy).sqrt();
        strict_math::acos(dot_product / (a_length * b_length))
    }

    /// `fetchHoleBridge`: David Eberly's algorithm for finding a bridge
    /// between a hole and the outer polygon.
    fn fetch_hole_bridge(&self, hole_node: usize, outer: usize) -> Option<usize> {
        let mut p = outer;
        let mut qx = f64::NEG_INFINITY;
        let hx = self.nx(hole_node);
        let hy = self.ny(hole_node);
        let mut connection: Option<usize> = None;
        // 1. find a segment intersected by a ray from the hole's leftmost point to the left;
        // segment's endpoint with lesser x will be potential connection point
        loop {
            let pn = self.next(p);
            if hy <= self.ny(p) && hy >= self.ny(pn) && self.ny(pn) != self.ny(p) {
                let x = self.nx(p)
                    + (hy - self.ny(p)) * (self.nx(pn) - self.nx(p)) / (self.ny(pn) - self.ny(p));
                if x <= hx && x > qx {
                    qx = x;
                    if x == hx {
                        if hy == self.ny(p) {
                            return Some(p);
                        }
                        if hy == self.ny(pn) {
                            return Some(pn);
                        }
                    }
                    connection = Some(if self.nx(p) < self.nx(pn) { p } else { pn });
                }
            }
            p = pn;
            if p == outer {
                break;
            }
        }
        let mut connection = connection?;
        if hx == qx {
            return Some(self.prev(connection));
        }
        // 2. look for points inside the triangle of hole point, segment intersection, and endpoint
        // its a valid connection iff there are no points found;
        // otherwise choose the point of the minimum angle with the ray as the connection point
        let stop = connection;
        let mx = self.nx(connection);
        let my = self.ny(connection);
        let mut tan_min = f64::INFINITY;
        p = connection;
        loop {
            let (px, py) = (self.nx(p), self.ny(p));
            if hx >= px
                && px >= mx
                && hx != px
                && point_in_ear(
                    px,
                    py,
                    if hy < my { hx } else { qx },
                    hy,
                    mx,
                    my,
                    if hy < my { qx } else { hx },
                    hy,
                )
            {
                let tan = (hy - py).abs() / (hx - px); // tangential
                if (tan < tan_min || (tan == tan_min && px > self.nx(connection)))
                    && self.is_locally_inside(p, hole_node)
                {
                    connection = p;
                    tan_min = tan;
                }
            }
            p = self.next(p);
            if p == stop {
                break;
            }
        }
        Some(connection)
    }

    /// `fetchLeftmost`: the left-most node of a ring.
    fn fetch_leftmost(&self, start: usize) -> usize {
        let mut node = start;
        let mut left_most = start;
        loop {
            if self.nx(node) < self.nx(left_most)
                || (self.nx(node) == self.nx(left_most) && self.ny(node) < self.ny(left_most))
            {
                left_most = node;
            }
            node = self.next(node);
            if node == start {
                return left_most;
            }
        }
    }

    // ------------------------------------------------------------ ear slicing

    fn make_triangle(
        &self,
        a: usize,
        ab: bool,
        b: usize,
        bc: bool,
        c: usize,
        ca: bool,
    ) -> Triangle {
        let (na, nb, nc) = (self.n(a), self.n(b), self.n(c));
        Triangle {
            encoded_x: [na.x, nb.x, nc.x],
            encoded_y: [na.y, nb.y, nc.y],
            x: [na.vx, nb.vx, nc.vx],
            y: [na.vy, nb.vy, nc.vy],
            edge_from_polygon: [ab, bc, ca],
        }
    }

    /// `earcutLinkedList`: the main ear slicing loop.
    fn earcut_linked_list(
        &mut self,
        mut curr_ear: Option<usize>,
        tessellation: &mut Vec<Triangle>,
        mut state: State,
        morton_optimized: bool,
        depth: i32,
    ) -> Result<(), GeoError> {
        'earcut: loop {
            let mut ear = match curr_ear {
                Some(e) if self.prev(e) != self.next(e) => e,
                _ => return Ok(()),
            };
            let mut stop = ear;
            // Iteratively slice ears
            loop {
                self.notify_monitor(state, depth, Some(ear), tessellation)?;
                let prev_node = self.prev(ear);
                let next_node = self.next(ear);
                // Determine whether the current triangle must be cut off.
                let is_reflex = area(
                    self.nx(prev_node),
                    self.ny(prev_node),
                    self.nx(ear),
                    self.ny(ear),
                    self.nx(next_node),
                    self.ny(next_node),
                ) >= 0.0;
                if !is_reflex && self.is_ear(ear, morton_optimized) {
                    // Compute if edges belong to the polygon
                    let ab = self.n(prev_node).is_next_edge_from_polygon;
                    let bc = self.n(ear).is_next_edge_from_polygon;
                    let ca = self.is_edge_from_polygon(prev_node, next_node, morton_optimized);
                    tessellation.push(self.make_triangle(prev_node, ab, ear, bc, next_node, ca));
                    // Remove the ear node.
                    self.remove_node(ear, ca);
                    // Skipping to the next node leaves fewer slither triangles.
                    ear = self.next(next_node);
                    stop = self.next(next_node);
                    // Java's `continue` re-tests the do-while condition.
                    if self.prev(ear) != self.next(ear) {
                        continue;
                    }
                    break;
                }
                ear = next_node;
                // If the whole polygon has been iterated over and no more ears can be found.
                if ear == stop {
                    match state {
                        State::Init => {
                            // try filtering points and slicing again
                            curr_ear = Some(self.filter_points(ear, None));
                            state = State::Cure;
                            continue 'earcut;
                        }
                        State::Cure => {
                            // if this didn't work, try curing all small self-intersections locally
                            curr_ear = Some(self.cure_local_intersections(
                                ear,
                                tessellation,
                                morton_optimized,
                            ));
                            state = State::Split;
                            continue 'earcut;
                        }
                        State::Split => {
                            // as a last resort, try splitting the remaining polygon into two
                            if !self.split_earcut(ear, tessellation, morton_optimized, depth + 1)? {
                                // we could not process all points. Tessellation failed
                                let status = format!("{}[FAILED]", state.name());
                                self.notify_monitor_status(&status, Some(ear), tessellation)?;
                                return Err(illegal_tessellate());
                            }
                        }
                    }
                    break;
                }
                if self.prev(ear) == self.next(ear) {
                    break;
                }
            }
            return Ok(());
        }
    }

    /// `isEar`: whether a node forms a valid ear with its neighbours.
    fn is_ear(&self, ear: usize, morton_optimized: bool) -> bool {
        if morton_optimized {
            return self.morton_is_ear(ear);
        }
        let (ep, en) = (self.prev(ear), self.next(ear));
        let (ax, ay, bx, by, cx, cy) = (
            self.nx(ep),
            self.ny(ep),
            self.nx(ear),
            self.ny(ear),
            self.nx(en),
            self.ny(en),
        );
        // make sure there aren't other points inside the potential ear
        let mut node = self.next(en);
        while node != ep {
            if point_in_ear(self.nx(node), self.ny(node), ax, ay, bx, by, cx, cy)
                && self.node_area(node) >= 0.0
            {
                return false;
            }
            node = self.next(node);
        }
        true
    }

    /// `area(node.previous, node, node.next)`.
    #[inline]
    fn node_area(&self, node: usize) -> f64 {
        let (p, n) = (self.prev(node), self.next(node));
        area(
            self.nx(p),
            self.ny(p),
            self.nx(node),
            self.ny(node),
            self.nx(n),
            self.ny(n),
        )
    }

    /// The z-order range of a bounding box of encoded coordinates.
    fn z_range(&self, xs: &[i32], ys: &[i32]) -> (i64, i64) {
        let min_tx = flip(*xs.iter().min().expect("non-empty"));
        let min_ty = flip(*ys.iter().min().expect("non-empty"));
        let max_tx = flip(*xs.iter().max().expect("non-empty"));
        let max_ty = flip(*ys.iter().max().expect("non-empty"));
        (
            bit_util::interleave(min_tx, min_ty),
            bit_util::interleave(max_tx, max_ty),
        )
    }

    /// `mortonIsEar`: `isEar` that only visits nodes in the triangle's
    /// z-order range.
    fn morton_is_ear(&self, ear: usize) -> bool {
        let (ep, en) = (self.prev(ear), self.next(ear));
        // triangle bbox (flip the bits so negative encoded values are < positive encoded values)
        let (min_z, max_z) = self.z_range(
            &[self.n(ep).x, self.n(ear).x, self.n(en).x],
            &[self.n(ep).y, self.n(ear).y, self.n(en).y],
        );
        let (ax, ay, bx, by, cx, cy) = (
            self.nx(ep),
            self.ny(ep),
            self.nx(ear),
            self.ny(ear),
            self.nx(en),
            self.ny(en),
        );
        let (ep_idx, en_idx) = (self.n(ep).idx, self.n(en).idx);
        let blocks = |q: usize| {
            self.n(q).idx != ep_idx
                && self.n(q).idx != en_idx
                && point_in_ear(self.nx(q), self.ny(q), ax, ay, bx, by, cx, cy)
                && self.node_area(q) >= 0.0
        };
        // look for points inside the triangle in both directions
        let mut p = self.n(ear).previous_z;
        let mut n = self.n(ear).next_z;
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(ucmp(self.n(pp).morton) >= ucmp(min_z) && ucmp(self.n(nn).morton) <= ucmp(max_z)) {
                break;
            }
            if blocks(pp) {
                return false;
            }
            p = self.n(pp).previous_z;
            if blocks(nn) {
                return false;
            }
            n = self.n(nn).next_z;
        }
        // first look for points inside the triangle in decreasing z-order
        while let Some(pp) = p {
            if ucmp(self.n(pp).morton) < ucmp(min_z) {
                break;
            }
            if blocks(pp) {
                return false;
            }
            p = self.n(pp).previous_z;
        }
        // then look for points in increasing z-order
        while let Some(nn) = n {
            if ucmp(self.n(nn).morton) > ucmp(max_z) {
                break;
            }
            if blocks(nn) {
                return false;
            }
            n = self.n(nn).next_z;
        }
        true
    }

    /// `cureLocalIntersections`: removes small local self-intersections.
    fn cure_local_intersections(
        &mut self,
        mut start_node: usize,
        tessellation: &mut Vec<Triangle>,
        morton_optimized: bool,
    ) -> usize {
        let mut node = start_node;
        loop {
            let next_node = self.next(node);
            let a = self.prev(node);
            let b = self.next(next_node);
            // a self-intersection where edge (v[i-1],v[i]) intersects (v[i+1],v[i+2])
            if !self.is_vertex_equals(a, b)
                && lines_intersect(
                    self.nx(a),
                    self.ny(a),
                    self.nx(node),
                    self.ny(node),
                    self.nx(next_node),
                    self.ny(next_node),
                    self.nx(b),
                    self.ny(b),
                )
                && self.is_locally_inside(a, b)
                && self.is_locally_inside(b, a)
                // this call is expensive so do it last
                && !self.is_intersecting_polygon(a, self.nx(a), self.ny(a), self.nx(b), self.ny(b))
            {
                // compute edges from polygon
                let ab = if self.next(a) == node {
                    self.n(a).is_next_edge_from_polygon
                } else {
                    self.is_edge_from_polygon(a, node, morton_optimized)
                };
                let bc = if self.next(node) == b {
                    self.n(node).is_next_edge_from_polygon
                } else {
                    self.is_edge_from_polygon(node, b, morton_optimized)
                };
                let ca = if self.next(b) == a {
                    self.n(b).is_next_edge_from_polygon
                } else {
                    self.is_edge_from_polygon(a, b, morton_optimized)
                };
                let tri = self.make_triangle(a, ab, node, bc, b, ca);
                tessellation.push(tri);
                // Return the triangulated vertices to the tessellation
                // (Java adds this triangle twice too).
                tessellation.push(tri);
                // remove two nodes involved
                self.remove_node(node, ca);
                let node_next = self.next(node);
                self.remove_node(node_next, ca);
                node = b;
                start_node = b;
            }
            node = self.next(node);
            if node == start_node {
                return node;
            }
        }
    }

    /// `splitEarcut`: splits the polygon along a valid diagonal and
    /// triangulates each side. Returns whether it split.
    fn split_earcut(
        &mut self,
        start: usize,
        tessellation: &mut Vec<Triangle>,
        morton_optimized: bool,
        depth: i32,
    ) -> Result<bool, GeoError> {
        // Search for a valid diagonal that divides the polygon into two.
        let mut search_node = start;
        loop {
            let next_node = self.next(search_node);
            let mut diagonal = self.next(next_node);
            while diagonal != self.prev(search_node) {
                if self.n(search_node).idx != self.n(diagonal).idx
                    && self.is_valid_diagonal(search_node, diagonal)
                {
                    // Split the polygon into two at the point of the diagonal
                    let from_polygon =
                        self.is_edge_from_polygon(search_node, diagonal, morton_optimized);
                    let mut split_node = self.split_polygon(search_node, diagonal, from_polygon);
                    // Filter the resulting polygon.
                    search_node = self.filter_points(search_node, Some(self.next(search_node)));
                    split_node = self.filter_points(split_node, Some(self.next(split_node)));
                    // Attempt to earcut both of the resulting polygons
                    if morton_optimized {
                        self.sort_by_morton_with_reset(search_node);
                        self.sort_by_morton_with_reset(split_node);
                    }
                    self.notify_monitor_split(depth, search_node, split_node)?;
                    self.earcut_linked_list(
                        Some(search_node),
                        tessellation,
                        State::Init,
                        morton_optimized,
                        depth,
                    )?;
                    self.earcut_linked_list(
                        Some(split_node),
                        tessellation,
                        State::Init,
                        morton_optimized,
                        depth,
                    )?;
                    self.notify_monitor_split_end(depth);
                    // Finish the iterative search
                    return Ok(true);
                }
                diagonal = self.next(diagonal);
            }
            search_node = self.next(search_node);
            if search_node == start {
                break;
            }
        }
        // if there is some area left, we failed
        Ok(self.signed_area(start, start) == 0.0)
    }

    // ------------------------------------------------------------ intersection checks

    /// `checkIntersection`: throws on a polygon that intersects itself.
    fn check_intersection(&self, a: usize, is_morton: bool) -> Result<(), GeoError> {
        let mut next = self.next(a);
        loop {
            let mut inner_next = self.next(next);
            if is_morton {
                self.morton_check_intersection(next, inner_next)?;
            } else {
                loop {
                    self.check_intersection_point(next, inner_next)?;
                    inner_next = self.next(inner_next);
                    if inner_next == self.prev(next) {
                        break;
                    }
                }
            }
            next = self.next(next);
            if next == self.prev(a) {
                return Ok(());
            }
        }
    }

    /// `mortonCheckIntersection`.
    fn morton_check_intersection(&self, a: usize, b: usize) -> Result<(), GeoError> {
        let an = self.next(a);
        let (min_z, max_z) =
            self.z_range(&[self.n(a).x, self.n(an).x], &[self.n(a).y, self.n(an).y]);
        let mut p = self.n(b).previous_z;
        let mut n = self.n(b).next_z;
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(ucmp(self.n(pp).morton) >= ucmp(min_z) && ucmp(self.n(nn).morton) <= ucmp(max_z)) {
                break;
            }
            self.check_intersection_point(pp, a)?;
            p = self.n(pp).previous_z;
            self.check_intersection_point(nn, a)?;
            n = self.n(nn).next_z;
        }
        while let Some(pp) = p {
            if ucmp(self.n(pp).morton) < ucmp(min_z) {
                break;
            }
            self.check_intersection_point(pp, a)?;
            p = self.n(pp).previous_z;
        }
        while let Some(nn) = n {
            if ucmp(self.n(nn).morton) > ucmp(max_z) {
                break;
            }
            self.check_intersection_point(nn, a)?;
            n = self.n(nn).next_z;
        }
        Ok(())
    }

    /// `checkIntersectionPoint`.
    fn check_intersection_point(&self, a: usize, b: usize) -> Result<(), GeoError> {
        if a == b {
            return Ok(());
        }
        let (an, bn) = (self.next(a), self.next(b));
        let (ax, ay, anx, any) = (self.nx(a), self.ny(a), self.nx(an), self.ny(an));
        let (bx, by, bnx, bny) = (self.nx(b), self.ny(b), self.nx(bn), self.ny(bn));
        use super::{java_max as mx, java_min as mn};
        if mx(ay, any) <= mn(by, bny)
            || mn(ay, any) >= mx(by, bny)
            || mx(ax, anx) <= mn(bx, bnx)
            || mn(ax, anx) >= mx(bx, bnx)
        {
            return Ok(());
        }
        if GeoUtils::line_crosses_line(ax, ay, anx, any, bx, by, bnx, bny) {
            // Line AB represented as a1x + b1y = c1
            let a1 = any - ay;
            let b1 = ax - anx;
            let c1 = a1 * ax + b1 * ay;
            // Line CD represented as a2x + b2y = c2
            let a2 = bny - by;
            let b2 = bx - bnx;
            let c2 = a2 * bx + b2 * by;
            let determinant = a1 * b2 - a2 * b1;
            let x = (b2 * c1 - b1 * c2) / determinant;
            let y = (a1 * c2 - a2 * c1) / determinant;
            return Err(GeoError::illegal(format!(
                "Polygon self-intersection at lat={} lon={}",
                java_double_string(y),
                java_double_string(x)
            )));
        }
        if self.n(a).is_next_edge_from_polygon
            && self.n(b).is_next_edge_from_polygon
            && GeoUtils::line_overlap_line(ax, ay, anx, any, bx, by, bnx, bny)
        {
            return Err(GeoError::illegal(format!(
                "Polygon ring self-intersection at lat={} lon={}",
                java_double_string(ay),
                java_double_string(ax)
            )));
        }
        Ok(())
    }

    /// `isEdgeFromPolygon`: whether the edge a-b overlaps a polygon edge.
    fn is_edge_from_polygon(&self, a: usize, b: usize, is_morton: bool) -> bool {
        if is_morton {
            return self.is_morton_edge_from_polygon(a, b);
        }
        let mut next = a;
        loop {
            if let Some(r) = self.edge_match(next, a, b) {
                return r;
            }
            next = self.next(next);
            if next == a {
                return false;
            }
        }
    }

    /// The two checks `isEdgeFromPolygon` makes at each node.
    #[inline]
    fn edge_match(&self, q: usize, a: usize, b: usize) -> Option<bool> {
        let (qn, qp) = (self.next(q), self.prev(q));
        if self.is_point_in_line(q, qn, a) && self.is_point_in_line(q, qn, b) {
            return Some(self.n(q).is_next_edge_from_polygon);
        }
        if self.is_point_in_line(q, qp, a) && self.is_point_in_line(q, qp, b) {
            return Some(self.n(qp).is_next_edge_from_polygon);
        }
        None
    }

    /// `isMortonEdgeFromPolygon`.
    fn is_morton_edge_from_polygon(&self, a: usize, b: usize) -> bool {
        let (min_z, max_z) = self.z_range(&[self.n(a).x, self.n(b).x], &[self.n(a).y, self.n(b).y]);
        let mut p = self.n(a).previous_z;
        let mut n = self.n(a).next_z;
        while let (Some(pp), Some(nn)) = (p, n) {
            if !(ucmp(self.n(pp).morton) >= ucmp(min_z) && ucmp(self.n(nn).morton) <= ucmp(max_z)) {
                break;
            }
            if let Some(r) = self.edge_match(pp, a, b) {
                return r;
            }
            p = self.n(pp).previous_z;
            if let Some(r) = self.edge_match(nn, a, b) {
                return r;
            }
            n = self.n(nn).next_z;
        }
        while let Some(pp) = p {
            if ucmp(self.n(pp).morton) < ucmp(min_z) {
                break;
            }
            if let Some(r) = self.edge_match(pp, a, b) {
                return r;
            }
            p = self.n(pp).previous_z;
        }
        while let Some(nn) = n {
            if ucmp(self.n(nn).morton) > ucmp(max_z) {
                break;
            }
            if let Some(r) = self.edge_match(nn, a, b) {
                return r;
            }
            n = self.n(nn).next_z;
        }
        false
    }

    #[inline]
    fn is_point_in_line(&self, a: usize, b: usize, point: usize) -> bool {
        self.is_point_in_line_xy(a, b, self.nx(point), self.ny(point))
    }

    /// `isPointInLine(a, b, lon, lat)`: the point is on segment a-b.
    fn is_point_in_line_xy(&self, a: usize, b: usize, lon: f64, lat: f64) -> bool {
        let (ax, ay, bx, by) = (self.nx(a), self.ny(a), self.nx(b), self.ny(b));
        let dxc = lon - ax;
        let dyc = lat - ay;
        let dxl = bx - ax;
        let dyl = by - ay;
        if dxc * dyl - dyc * dxl == 0.0 {
            if dxl.abs() >= dyl.abs() {
                return if dxl > 0.0 {
                    ax <= lon && lon <= bx
                } else {
                    bx <= lon && lon <= ax
                };
            } else {
                return if dyl > 0.0 {
                    ay <= lat && lat <= by
                } else {
                    by <= lat && lat <= ay
                };
            }
        }
        false
    }

    /// `splitPolygon`: links two vertices with a bridge; returns the copy of
    /// `b`.
    fn split_polygon(&mut self, a: usize, b: usize, edge_from_polygon: bool) -> usize {
        let a2 = self.nodes.len();
        self.nodes.push(self.nodes[a]);
        let b2 = self.nodes.len();
        self.nodes.push(self.nodes[b]);
        let an = self.next(a);
        let bp = self.prev(b);

        self.nodes[a].next = b;
        self.nodes[a].is_next_edge_from_polygon = edge_from_polygon;
        self.nodes[a].next_z = Some(b);
        self.nodes[b].previous = a;
        self.nodes[b].previous_z = Some(a);
        self.nodes[a2].next = an;
        self.nodes[a2].next_z = Some(an);
        self.nodes[an].previous = a2;
        self.nodes[an].previous_z = Some(a2);
        self.nodes[b2].next = a2;
        self.nodes[b2].is_next_edge_from_polygon = edge_from_polygon;
        self.nodes[b2].next_z = Some(a2);
        self.nodes[a2].previous = b2;
        self.nodes[a2].previous_z = Some(b2);
        self.nodes[bp].next = b2;
        self.nodes[bp].next_z = Some(b2);
        b2
    }

    /// `isValidDiagonal`: the diagonal a-b lies within the polygon.
    fn is_valid_diagonal(&self, a: usize, b: usize) -> bool {
        let (an, ap, bn, bp) = (self.next(a), self.prev(a), self.next(b), self.prev(b));
        if self.n(an).idx == self.n(b).idx
            || self.n(ap).idx == self.n(b).idx
            // check next edges are locally visible
            || !self.is_locally_inside(ap, b)
            || !self.is_locally_inside(bn, a)
            // check polygons are CCW in both sides
            || !self.is_cw_polygon(a, b)
            || !self.is_cw_polygon(b, a)
        {
            return false;
        }
        if self.is_vertex_equals(a, b) {
            return true;
        }
        let (ax, ay, bx, by) = (self.nx(a), self.ny(a), self.nx(b), self.ny(b));
        self.is_locally_inside(a, b)
            && self.is_locally_inside(b, a)
            && self.middle_insert(a, ax, ay, bx, by)
            // make sure we don't introduce collinear lines
            && area(self.nx(ap), self.ny(ap), ax, ay, bx, by) != 0.0
            && area(ax, ay, bx, by, self.nx(bn), self.ny(bn)) != 0.0
            && area(self.nx(an), self.ny(an), ax, ay, bx, by) != 0.0
            && area(ax, ay, bx, by, self.nx(bp), self.ny(bp)) != 0.0
            // this call is expensive so do it last
            && !self.is_intersecting_polygon(a, ax, ay, bx, by)
    }

    /// `isCWPolygon`.
    fn is_cw_polygon(&self, start: usize, end: usize) -> bool {
        self.signed_area(start, end) < 0.0
    }

    /// `signedArea`: the signed area between node start and node end.
    fn signed_area(&self, start: usize, end: usize) -> f64 {
        let mut next = start;
        let mut winding_sum = 0.0;
        let (ex, ey) = (self.nx(end), self.ny(end));
        loop {
            let nn = self.next(next);
            winding_sum += area(
                self.nx(next),
                self.ny(next),
                self.nx(nn),
                self.ny(nn),
                ex,
                ey,
            );
            next = nn;
            if self.next(next) == end {
                return winding_sum;
            }
        }
    }

    /// `isLocallyInside`.
    fn is_locally_inside(&self, a: usize, b: usize) -> bool {
        let (ap, an) = (self.prev(a), self.next(a));
        let (ax, ay, bx, by) = (self.nx(a), self.ny(a), self.nx(b), self.ny(b));
        let area_a = area(self.nx(ap), self.ny(ap), ax, ay, self.nx(an), self.ny(an));
        if area_a == 0.0 {
            // parallel
            false
        } else if area_a < 0.0 {
            // if a is cw
            area(ax, ay, bx, by, self.nx(an), self.ny(an)) >= 0.0
                && area(ax, ay, self.nx(ap), self.ny(ap), bx, by) >= 0.0
        } else {
            // ccw
            area(ax, ay, bx, by, self.nx(ap), self.ny(ap)) <= 0.0
                || area(ax, ay, self.nx(an), self.ny(an), bx, by) <= 0.0
        }
    }

    /// `middleInsert`: the diagonal's midpoint is inside the polygon.
    fn middle_insert(&self, start: usize, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        let mut node = start;
        let mut l_is_inside = false;
        let l_dx = (x0 + x1) / 2.0;
        let l_dy = (y0 + y1) / 2.0;
        loop {
            let next_node = self.next(node);
            let (nx, ny, mx, my) = (
                self.nx(node),
                self.ny(node),
                self.nx(next_node),
                self.ny(next_node),
            );
            if (ny > l_dy) != (my > l_dy) && l_dx < (mx - nx) * (l_dy - ny) / (my - ny) + nx {
                l_is_inside = !l_is_inside;
            }
            node = next_node;
            if node == start {
                return l_is_inside;
            }
        }
    }

    /// `isIntersectingPolygon`: the diagonal crosses a polygon edge.
    fn is_intersecting_polygon(&self, start: usize, x0: f64, y0: f64, x1: f64, y1: f64) -> bool {
        let mut node = start;
        loop {
            let next_node = self.next(node);
            if !self.is_vertex_equals_xy(node, x0, y0)
                && !self.is_vertex_equals_xy(node, x1, y1)
                && lines_intersect(
                    self.nx(node),
                    self.ny(node),
                    self.nx(next_node),
                    self.ny(next_node),
                    x0,
                    y0,
                    x1,
                    y1,
                )
            {
                return true;
            }
            node = next_node;
            if node == start {
                return false;
            }
        }
    }

    // ------------------------------------------------------------ z-order

    /// `sortByMortonWithReset`: re-seeds the z links from the ring, then
    /// sorts.
    fn sort_by_morton_with_reset(&mut self, start: usize) {
        let mut next = start;
        loop {
            self.nodes[next].previous_z = Some(self.prev(next));
            self.nodes[next].next_z = Some(self.next(next));
            next = self.next(next);
            if next == start {
                break;
            }
        }
        self.sort_by_morton(start);
    }

    /// `sortByMorton`: interlinks the nodes in z-order.
    fn sort_by_morton(&mut self, start: usize) {
        if let Some(pz) = self.nodes[start].previous_z {
            self.nodes[pz].next_z = None;
        }
        self.nodes[start].previous_z = None;
        // Sort the generated ring using Z ordering.
        self.tatham_sort(start);
    }

    /// `tathamSort`: Simon Tatham's doubly-linked-list merge sort on the z
    /// links.
    fn tatham_sort(&mut self, list: usize) {
        let mut list = Some(list);
        let mut in_size = 1usize;
        loop {
            let mut p = list;
            list = None;
            let mut tail: Option<usize> = None;
            // count number of merges in this pass
            let mut num_merges = 0;
            while let Some(p_start) = p {
                num_merges += 1;
                // step 'insize' places along from p
                let mut q = Some(p_start);
                let mut p_size = 0;
                while p_size < in_size {
                    match q {
                        Some(qq) => {
                            q = self.nodes[qq].next_z;
                            p_size += 1;
                        }
                        None => break,
                    }
                }
                // if q hasn't fallen off end, we have two lists to merge
                let mut q_size = in_size;
                // now we have two lists; merge
                let mut p_cur = Some(p_start);
                while p_size > 0 || (q_size > 0 && q.is_some()) {
                    let take_p = match (p_cur, q) {
                        (Some(pc), Some(qc)) if p_size != 0 && q_size != 0 => {
                            ucmp(self.nodes[pc].morton) <= ucmp(self.nodes[qc].morton)
                        }
                        _ => p_size != 0,
                    };
                    let e = if take_p {
                        let e = p_cur.expect("p_size > 0");
                        p_cur = self.nodes[e].next_z;
                        p_size -= 1;
                        e
                    } else {
                        let e = q.expect("q present");
                        q = self.nodes[e].next_z;
                        q_size -= 1;
                        e
                    };
                    match tail {
                        Some(t) => self.nodes[t].next_z = Some(e),
                        None => list = Some(e),
                    }
                    // maintain reverse pointers
                    self.nodes[e].previous_z = tail;
                    tail = Some(e);
                }
                // now p has stepped 'insize' places along, and q has too
                p = q;
            }
            if let Some(t) = tail {
                self.nodes[t].next_z = None;
            }
            in_size *= 2;
            if num_merges <= 1 {
                break;
            }
        }
    }

    // ------------------------------------------------------------ list surgery

    /// `filterPoints`: removes duplicate and collinear points between
    /// `start` and `end`; returns the (possibly moved) end.
    fn filter_points(&mut self, start: usize, end: Option<usize>) -> usize {
        let mut end = end.unwrap_or(start);
        let mut node = start;
        loop {
            let mut continue_iteration = false;
            let next_node = self.next(node);
            let prev_node = self.prev(node);
            // we can filter points when:
            // 1. they are the same
            // 2.- each one starts and ends in each other
            // 3.- they are collinear and both edges have the same value in .isNextEdgeFromPolygon
            // 4.-  they are collinear and second edge returns over the first edge
            if self.is_vertex_equals(node, next_node)
                || self.is_vertex_equals(prev_node, next_node)
                || ((self.n(prev_node).is_next_edge_from_polygon
                    == self.n(node).is_next_edge_from_polygon
                    || self.is_point_in_line_xy(
                        prev_node,
                        node,
                        self.nx(next_node),
                        self.ny(next_node),
                    ))
                    && area(
                        self.nx(prev_node),
                        self.ny(prev_node),
                        self.nx(node),
                        self.ny(node),
                        self.nx(next_node),
                        self.ny(next_node),
                    ) == 0.0)
            {
                // Remove the node
                let from_polygon = self.n(prev_node).is_next_edge_from_polygon;
                self.remove_node(node, from_polygon);
                node = prev_node;
                end = prev_node;
                if node == next_node {
                    break;
                }
                continue_iteration = true;
            } else {
                node = next_node;
            }
            if !(continue_iteration || node != end) {
                break;
            }
        }
        end
    }

    /// `insertNode`: creates a node and links it after `last_node`.
    fn insert_node(
        &mut self,
        x: &[f64],
        y: &[f64],
        index: i32,
        vertex_index: usize,
        last_node: Option<usize>,
        is_geo: bool,
    ) -> Result<usize, GeoError> {
        let (vx, vy) = (x[vertex_index], y[vertex_index]);
        // casting to float is safe as original values for non-geo are represented as floats
        let ey = if is_geo {
            GeoEncodingUtils::encode_latitude(vy)?
        } else {
            XYEncodingUtils::encode(vy as f32)?
        };
        let ex = if is_geo {
            GeoEncodingUtils::encode_longitude(vx)?
        } else {
            XYEncodingUtils::encode(vx as f32)?
        };
        let node = self.nodes.len();
        let mut n = Node {
            idx: index,
            vx,
            vy,
            x: ex,
            y: ey,
            morton: bit_util::interleave(flip(ex), flip(ey)),
            previous: node,
            next: node,
            previous_z: Some(node),
            next_z: Some(node),
            is_next_edge_from_polygon: true,
        };
        match last_node {
            None => self.nodes.push(n),
            Some(last) => {
                let last_next = self.next(last);
                let last_next_z = self.n(last).next_z.expect("a fresh ring is z-linked");
                n.next = last_next;
                n.next_z = Some(last_next);
                n.previous = last;
                n.previous_z = Some(last);
                self.nodes.push(n);
                self.nodes[last_next].previous = node;
                self.nodes[last_next_z].previous_z = Some(node);
                self.nodes[last].next = node;
                self.nodes[last].next_z = Some(node);
            }
        }
        Ok(node)
    }

    /// `removeNode`: unlinks a node (whose own links are left as they were).
    fn remove_node(&mut self, node: usize, edge_from_polygon: bool) {
        let (prev, next) = (self.prev(node), self.next(node));
        self.nodes[next].previous = prev;
        self.nodes[prev].next = next;
        self.nodes[prev].is_next_edge_from_polygon = edge_from_polygon;
        let (pz, nz) = (self.n(node).previous_z, self.n(node).next_z);
        if let Some(pz) = pz {
            self.nodes[pz].next_z = nz;
        }
        if let Some(nz) = nz {
            self.nodes[nz].previous_z = pz;
        }
    }

    #[inline]
    fn is_vertex_equals(&self, a: usize, b: usize) -> bool {
        self.is_vertex_equals_xy(a, self.nx(b), self.ny(b))
    }

    #[inline]
    fn is_vertex_equals_xy(&self, a: usize, x: f64, y: f64) -> bool {
        self.nx(a) == x && self.ny(a) == y
    }

    // ------------------------------------------------------------ monitor

    /// `getPoints`: the ring as `Point(lat=y, lon=x)`s (validated, as Java's
    /// constructor validates them).
    fn get_points(&self, start: usize) -> Result<Vec<Point>, GeoError> {
        let mut points = Vec::new();
        let mut node = start;
        loop {
            points.push(Point::new(self.ny(node), self.nx(node))?);
            node = self.next(node);
            if node == start {
                return Ok(points);
            }
        }
    }

    fn notify_monitor_split(
        &mut self,
        depth: i32,
        search_node: usize,
        diagonal_node: usize,
    ) -> Result<(), GeoError> {
        if self.monitor.is_some() {
            let left = self.get_points(search_node)?;
            let right = self.get_points(diagonal_node)?;
            let status = format!("SPLIT[{depth}]");
            if let Some(m) = self.monitor.as_mut() {
                m.start_split(&status, &left, &right);
            }
        }
        Ok(())
    }

    fn notify_monitor_split_end(&mut self, depth: i32) {
        if let Some(m) = self.monitor.as_mut() {
            m.end_split(&format!("SPLIT[{depth}]"));
        }
    }

    fn notify_monitor(
        &mut self,
        state: State,
        depth: i32,
        start: Option<usize>,
        tessellation: &[Triangle],
    ) -> Result<(), GeoError> {
        if self.monitor.is_some() {
            let status = if depth == 0 {
                state.name().to_string()
            } else {
                format!("{}[{depth}]", state.name())
            };
            self.notify_monitor_status(&status, start, tessellation)?;
        }
        Ok(())
    }

    fn notify_monitor_status(
        &mut self,
        status: &str,
        start: Option<usize>,
        tessellation: &[Triangle],
    ) -> Result<(), GeoError> {
        if self.monitor.is_some() {
            let points = match start {
                Some(s) => Some(self.get_points(s)?),
                None => None,
            };
            if let Some(m) = self.monitor.as_mut() {
                m.current_state(status, points.as_deref(), tessellation);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square() -> Polygon {
        Polygon::new(
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn square_is_two_triangles() {
        let t = tessellate(&square(), true).unwrap();
        assert_eq!(t.len(), 2);
        for tri in &t {
            for v in 0..3 {
                assert!(tri.x(v) == 0.0 || tri.x(v) == 1.0);
                assert!(tri.y(v) == 0.0 || tri.y(v) == 1.0);
                assert_eq!(
                    tri.encoded_x(v),
                    GeoEncodingUtils::encode_longitude(tri.x(v)).unwrap()
                );
                assert_eq!(
                    tri.encoded_y(v),
                    GeoEncodingUtils::encode_latitude(tri.y(v)).unwrap()
                );
            }
        }
        assert!(t[0].to_string().contains("[true]"));
        // each triangle has two polygon edges and the shared diagonal
        let edges: usize = t
            .iter()
            .map(|tri| (0..3).filter(|&e| tri.is_edge_from_polygon(e)).count())
            .sum();
        assert_eq!(edges, 4);
    }

    #[test]
    fn degenerate_inputs_fail_like_java() {
        let line = Polygon::new(&[0.0, 1.0, 2.0, 0.0], &[0.0, 1.0, 2.0, 0.0], vec![]).unwrap();
        assert_eq!(
            tessellate(&line, false).unwrap_err().to_string(),
            "at least three non-collinear points required"
        );
        let bow = Polygon::new(
            &[0.0, 1.0, 0.0, 1.0, 0.0],
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            vec![],
        )
        .unwrap();
        assert!(tessellate(&bow, true)
            .unwrap_err()
            .to_string()
            .starts_with("Polygon self-intersection at lat="));
        let xy = XYPolygon::new(
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            vec![],
        )
        .unwrap();
        assert_eq!(tessellate_xy(&xy, true).unwrap().len(), 2);
        assert!(lines_intersect(0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0));
    }

    #[derive(Default)]
    struct Recorder {
        states: Vec<String>,
        splits: Vec<String>,
    }

    impl Monitor for Recorder {
        fn current_state(
            &mut self,
            status: &str,
            points: Option<&[Point]>,
            tessellation: &[Triangle],
        ) {
            self.states.push(format!(
                "{status}:{}:{}",
                points.map_or(0, <[Point]>::len),
                tessellation.len()
            ));
        }
        fn start_split(&mut self, status: &str, l: &[Point], r: &[Point]) {
            self.splits
                .push(format!("start {status} {} {}", l.len(), r.len()));
        }
        fn end_split(&mut self, status: &str) {
            self.splits.push(format!("end {status}"));
        }
    }

    #[test]
    fn monitor_sees_every_step() {
        let mut rec = Recorder::default();
        let t = tessellate_with_monitor(&square(), false, Some(&mut rec)).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(rec.states.first().unwrap(), "INIT:4:0");
        assert_eq!(rec.states.last().unwrap(), "COMPLETED:0:2");
        let mut rec = Recorder::default();
        let xy = XYPolygon::new(
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 1.0, 0.0],
            vec![],
        )
        .unwrap();
        tessellate_xy_with_monitor(&xy, false, Some(&mut rec)).unwrap();
        assert_eq!(rec.states.last().unwrap(), "COMPLETED:0:2");
    }

    #[test]
    fn monitor_sees_splits() {
        // Four holes sharing one vertex: the SPLIT pass runs (and, as in
        // Java, cannot finish).
        let hole = |lats: [f64; 4], lons: [f64; 4]| Polygon::new(&lats, &lons, vec![]).unwrap();
        let p = Polygon::new(
            &[0.0, 0.0, 10.0, 10.0, 0.0],
            &[0.0, 10.0, 10.0, 0.0, 0.0],
            vec![
                hole([5.0, 5.0, 6.0, 5.0], [5.0, 7.0, 7.0, 5.0]),
                hole([5.0, 7.0, 7.0, 5.0], [5.0, 5.0, 4.0, 5.0]),
                hole([5.0, 5.0, 4.0, 5.0], [5.0, 3.0, 3.0, 5.0]),
                hole([5.0, 3.0, 3.0, 5.0], [5.0, 5.0, 6.0, 5.0]),
            ],
        )
        .unwrap();
        let mut rec = Recorder::default();
        let err = tessellate_with_monitor(&p, false, Some(&mut rec)).unwrap_err();
        assert_eq!(
            err.to_string(),
            "Unable to Tessellate shape. Possible malformed shape detected."
        );
        assert!(rec.splits.iter().any(|s| s.starts_with("start SPLIT[1]")));
        assert!(rec.states.iter().any(|s| s.starts_with("CURE")));
        assert!(rec.states.iter().any(|s| s.starts_with("INIT[1]")));
        assert!(rec.states.iter().any(|s| s.starts_with("SPLIT[FAILED]")));
        // A corpus polygon (holes on the south pole) whose split succeeds.
        let ring = |s: &str| -> (Vec<f64>, Vec<f64>) {
            s.split(';')
                .map(|p| {
                    let (a, b) = p.split_once(' ').unwrap();
                    (a.parse::<f64>().unwrap(), b.parse::<f64>().unwrap())
                })
                .unzip()
        };
        let (la, lo) = ring("-86.86737826154896 26.51454402962736;-84.4712724352173 26.09376343767987;-76.25355817827216 9.181299289049523;-76.60563758407736 7.214210164740096;-79.57433616804926 3.4897899309380236;-84.40820021900568 0.6857091029569009;-85.92640268234479 0.8782562408888506;-90.0 2.8263168243244827;-90.0 4.597102435947102;-90.0 5.6391846906056;-90.0 14.105011898137173;-90.0 19.778369747749295;-90.0 22.566639414502674;-86.86737826154896 26.51454402962736");
        let (h1a, h1o) = ring("-87.49341134820244 19.979778286956353;-90.0 18.46440296320393;-90.0 21.493435169262472;-87.49341134820244 19.979778286956353");
        let (h2a, h2o) = ring("-87.21810549623036 8.141296141436479;-89.65496009215225 8.142924166776647;-90.0 7.103439437222652;-90.0 6.5683028475352625;-90.0 6.1713462579135125;-88.16473787815741 4.731716690706579;-87.21810549623036 8.141296141436479");
        let p = Polygon::new(
            &la,
            &lo,
            vec![
                Polygon::new(&h1a, &h1o, vec![]).unwrap(),
                Polygon::new(&h2a, &h2o, vec![]).unwrap(),
            ],
        )
        .unwrap();
        let mut rec = Recorder::default();
        assert!(!tessellate_with_monitor(&p, false, Some(&mut rec))
            .unwrap()
            .is_empty());
        assert!(rec.splits.iter().any(|s| s == "end SPLIT[1]"));
        // A cartesian polygon off the globe cannot be reported as Points.
        let xy = XYPolygon::new(
            &[0.0, 1.0, 1.0, 0.0, 0.0],
            &[0.0, 0.0, 100.0, 100.0, 0.0],
            vec![],
        )
        .unwrap();
        let mut rec = Recorder::default();
        assert!(tessellate_xy_with_monitor(&xy, false, Some(&mut rec))
            .unwrap_err()
            .to_string()
            .starts_with("invalid latitude 100.0"));
    }
}
