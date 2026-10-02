//! `GeoComplexPolygon` (`org.apache.lucene.spatial3d.geom.GeoComplexPolygon`):
//! a polygon of any shape and size (with any holes), given as rings of
//! points and a test point known to be inside or outside. Membership is
//! decided by counting edge crossings along a path from the test point to
//! the point -- one axis-aligned plane, or two joined at an intersection
//! point -- with each axis's edges in an interval tree.
//!
//! Java's edges form a linked ring and the trees hold node objects; here
//! both live in vectors and refer to each other by index. The exceptions
//! Java throws from inside a traversal and catches a few frames up (to fall
//! back to another iterator or strategy) go through [`super::errors`].

#![allow(non_snake_case)]

use super::errors::{catch, raise};
use super::prelude::*;
use super::serializable::{read_boolean, read_count, write_boolean, write_int};
use super::shape::GeoPolygon;
use super::standard_objects::{read_point_array, write_point_array};
use super::vector::MINIMUM_RESOLUTION_SQUARED;
use super::xyz_bounds::XYZBounds;

const NEAR_EDGE_CUTOFF: f64 = -MINIMUM_RESOLUTION * 10000.0;
const HALF_PROPORTIONS: [f64; 1] = [0.5];
const DELTA_DISTANCE: f64 = MINIMUM_RESOLUTION;
const MAX_ITERATIONS: usize = 100;
const OFF_PLANE_AMOUNT: f64 = MINIMUM_RESOLUTION * 0.1;

/// An edge of a ring.
#[derive(Debug, Clone)]
struct Edge {
    start_point: GeoPoint,
    end_point: GeoPoint,
    notable_points: [GeoPoint; 2],
    start_plane: SidedPlane,
    end_plane: SidedPlane,
    backing_plane: SidedPlane,
    plane: Plane,
    plane_bounds: XYZBounds,
    /// The next edge of the ring (`next`).
    next: usize,
}

impl Edge {
    fn new(pm: &PlanetModel, start_point: GeoPoint, end_point: GeoPoint) -> Result<Edge> {
        let plane = Plane::from_vectors(&start_point, &end_point)?;
        let start_plane = SidedPlane::from_vectors(&end_point, &plane, &start_point)?;
        let end_plane = SidedPlane::from_vectors(&start_point, &plane, &end_point)?;
        let interpolation_point = plane
            .interpolate(pm, &start_point, &end_point, &HALF_PROPORTIONS)?
            .into_iter()
            .next()
            .expect("one proportion");
        let backing_plane =
            SidedPlane::from_normal(&interpolation_point, &interpolation_point, 0.0)?;
        let mut plane_bounds = XYZBounds::new();
        plane_bounds.add_point(&start_point);
        plane_bounds.add_point(&end_point);
        plane_bounds.add_plane(pm, &plane, &[&start_plane, &end_plane, &backing_plane]);
        Ok(Edge {
            notable_points: [start_point.clone(), end_point.clone()],
            start_point,
            end_point,
            start_plane,
            end_plane,
            backing_plane,
            plane,
            plane_bounds,
            next: 0,
        })
    }

    fn is_within(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane.evaluate_is_zero_xyz(x, y, z)
            && self.start_plane.is_within_xyz(x, y, z)
            && self.end_plane.is_within_xyz(x, y, z)
            && self.backing_plane.is_within_xyz(x, y, z)
    }
}

/// `EdgeIterator`: visits edges until `matches` says stop.
trait EdgeIterator {
    fn matches(&mut self, poly: &Inner, edge: usize) -> bool;
}

/// `CountingEdgeIterator`.
trait CountingEdgeIterator: EdgeIterator {
    fn crossing_count(&self) -> i32;
    fn is_on_edge(&self) -> bool;
}

/// A node of an interval tree.
#[derive(Debug, Clone)]
struct Node {
    edge: usize,
    low: f64,
    high: f64,
    left: Option<usize>,
    right: Option<usize>,
    max: f64,
}

/// `Tree`: edges by their extent along one axis.
#[derive(Debug, Clone)]
struct Tree {
    nodes: Vec<Node>,
    root: Option<usize>,
}

/// `Double.compare`.
fn java_double_compare(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    if a < b {
        return Ordering::Less;
    }
    if a > b {
        return Ordering::Greater;
    }
    let ab = super::jmath::double_to_long_bits(a);
    let bb = super::jmath::double_to_long_bits(b);
    ab.cmp(&bb)
}

#[derive(Debug, Clone, Copy)]
enum Axis {
    X,
    Y,
    Z,
}

impl Tree {
    fn new(edges: &[Edge], axis: Axis) -> Tree {
        let bound = |e: &Edge| -> (f64, f64) {
            let b = &e.plane_bounds;
            let (lo, hi) = match axis {
                Axis::X => (b.minimum_x(), b.maximum_x()),
                Axis::Y => (b.minimum_y(), b.maximum_y()),
                Axis::Z => (b.minimum_z(), b.maximum_z()),
            };
            // Every edge's bounds hold its two points.
            (lo.unwrap_or(f64::NAN), hi.unwrap_or(f64::NAN))
        };
        let mut nodes: Vec<Node> = edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let (low, high) = bound(e);
                Node {
                    edge: i,
                    low,
                    high,
                    left: None,
                    right: None,
                    max: high,
                }
            })
            .collect();
        nodes.sort_by(|l, r| {
            java_double_compare(l.low, r.low).then_with(|| java_double_compare(l.max, r.max))
        });
        let root = if nodes.is_empty() {
            None
        } else {
            {
                let last = nodes.len() as isize - 1;
                Self::create_tree(&mut nodes, 0, last)
            }
        };
        Tree { nodes, root }
    }

    fn create_tree(nodes: &mut [Node], low: isize, high: isize) -> Option<usize> {
        if low > high {
            return None;
        }
        let mid = ((low + high) as usize) >> 1;
        let left = Self::create_tree(nodes, low, mid as isize - 1);
        let right = Self::create_tree(nodes, mid as isize + 1, high);
        nodes[mid].left = left;
        nodes[mid].right = right;
        if let Some(l) = left {
            nodes[mid].max = max(nodes[mid].max, nodes[l].max);
        }
        if let Some(r) = right {
            nodes[mid].max = max(nodes[mid].max, nodes[r].max);
        }
        Some(mid)
    }

    fn traverse_value(&self, poly: &Inner, it: &mut dyn EdgeIterator, value: f64) -> bool {
        self.traverse(poly, it, value, value)
    }

    fn traverse(
        &self,
        poly: &Inner,
        it: &mut dyn EdgeIterator,
        min_value: f64,
        max_value: f64,
    ) -> bool {
        match self.root {
            None => true,
            Some(r) => self.traverse_node(poly, r, it, min_value, max_value),
        }
    }

    fn traverse_node(
        &self,
        poly: &Inner,
        n: usize,
        it: &mut dyn EdgeIterator,
        min_value: f64,
        max_value: f64,
    ) -> bool {
        let node = &self.nodes[n];
        if min_value <= node.max {
            // Does this node overlap?
            if min_value <= node.high && max_value >= node.low && !it.matches(poly, node.edge) {
                return false;
            }
            if let Some(l) = node.left {
                if !self.traverse_node(poly, l, it, min_value, max_value) {
                    return false;
                }
            }
            if let Some(r) = node.right {
                if max_value >= node.low && !self.traverse_node(poly, r, it, min_value, max_value) {
                    return false;
                }
            }
        }
        true
    }
}

/// The polygon's search structure, shared by its iterators.
#[derive(Debug, Clone)]
struct Inner {
    planet_model: Arc<PlanetModel>,
    edges: Vec<Edge>,
    x_tree: Tree,
    y_tree: Tree,
    z_tree: Tree,
}

/// A complex polygon.
#[derive(Debug, Clone)]
pub struct GeoComplexPolygon {
    planet_model: Arc<PlanetModel>,
    inner: Inner,
    points_list: Vec<Vec<GeoPoint>>,
    test_point1_in_set: bool,
    test_point1: GeoPoint,
    test_point1_fixed_y_plane: Plane,
    test_point1_fixed_y_above_plane: Option<Plane>,
    test_point1_fixed_y_below_plane: Option<Plane>,
    test_point1_fixed_x_plane: Plane,
    test_point1_fixed_x_above_plane: Option<Plane>,
    test_point1_fixed_x_below_plane: Option<Plane>,
    test_point1_fixed_z_plane: Plane,
    test_point1_fixed_z_above_plane: Option<Plane>,
    test_point1_fixed_z_below_plane: Option<Plane>,
    edge_points: Vec<GeoPoint>,
    shape_start_edges: Vec<usize>,
}

/// The above/below planes of a fixed-coordinate plane, `None` where they
/// would leave the planet (`NEAR_EDGE_CUTOFF`).
fn offset_plane(pm: &PlanetModel, base: &Plane, above: bool, axis: Axis) -> Option<Plane> {
    let p = Plane::offset(base, above);
    let (min_v, max_v) = match axis {
        Axis::X => (pm.minimum_x_value(), pm.maximum_x_value()),
        Axis::Y => (pm.minimum_y_value(), pm.maximum_y_value()),
        Axis::Z => (pm.minimum_z_value(), pm.maximum_z_value()),
    };
    if -p.D - max_v > NEAR_EDGE_CUTOFF || min_v + p.D > NEAR_EDGE_CUTOFF {
        None
    } else {
        Some(p)
    }
}

impl GeoComplexPolygon {
    /// `GeoComplexPolygon(planetModel, pointsList, testPoint,
    /// testPointInSet)`: `pointsList` are the rings (each implicitly
    /// closed), `testPoint` a surface point known to be inside
    /// (`test_point_in_set`) or outside.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        points_list: Vec<Vec<GeoPoint>>,
        test_point: GeoPoint,
        test_point_in_set: bool,
    ) -> Result<GeoComplexPolygon> {
        let pm = &**planet_model;
        let mut edge_points = Vec::with_capacity(points_list.len());
        let mut shape_start_edges = Vec::with_capacity(points_list.len());
        let mut all_edges: Vec<Edge> = Vec::new();
        for shape_points in &points_list {
            let Some(mut last_geo_point) = shape_points.last() else {
                return Err(Error::IndexOutOfBounds(
                    "Index -1 out of bounds for length 0".into(),
                ));
            };
            edge_points.push(last_geo_point.clone());
            let first_edge = all_edges.len();
            for this_geo_point in shape_points {
                let edge = Edge::new(pm, last_geo_point.clone(), this_geo_point.clone())?;
                if edge.is_within(test_point.x, test_point.y, test_point.z) {
                    return Err(illegal("Test point is on polygon edge: not allowed"));
                }
                let index = all_edges.len();
                if index > first_edge {
                    all_edges[index - 1].next = index;
                }
                all_edges.push(edge);
                last_geo_point = this_geo_point;
            }
            let last = all_edges.len() - 1;
            all_edges[last].next = first_edge;
            shape_start_edges.push(first_edge);
        }
        let inner = Inner {
            planet_model: planet_model.clone(),
            x_tree: Tree::new(&all_edges, Axis::X),
            y_tree: Tree::new(&all_edges, Axis::Y),
            z_tree: Tree::new(&all_edges, Axis::Z),
            edges: all_edges,
        };
        let fy = Plane::new(0.0, 1.0, 0.0, -test_point.y);
        let fx = Plane::new(1.0, 0.0, 0.0, -test_point.x);
        let fz = Plane::new(0.0, 0.0, 1.0, -test_point.z);
        Ok(GeoComplexPolygon {
            planet_model: planet_model.clone(),
            inner,
            points_list,
            test_point1_in_set: test_point_in_set,
            test_point1_fixed_y_above_plane: offset_plane(pm, &fy, true, Axis::Y),
            test_point1_fixed_y_below_plane: offset_plane(pm, &fy, false, Axis::Y),
            test_point1_fixed_x_above_plane: offset_plane(pm, &fx, true, Axis::X),
            test_point1_fixed_x_below_plane: offset_plane(pm, &fx, false, Axis::X),
            test_point1_fixed_z_above_plane: offset_plane(pm, &fz, true, Axis::Z),
            test_point1_fixed_z_below_plane: offset_plane(pm, &fz, false, Axis::Z),
            test_point1_fixed_y_plane: fy,
            test_point1_fixed_x_plane: fx,
            test_point1_fixed_z_plane: fz,
            test_point1: test_point,
            edge_points,
            shape_start_edges,
        })
    }

    /// `GeoComplexPolygon(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoComplexPolygon> {
        let count = read_count(input)?;
        let mut points_list = Vec::with_capacity(count.min(1 << 16));
        for _ in 0..count {
            points_list.push(read_point_array(input)?);
        }
        let test_point = GeoPoint::read(input)?;
        let in_set = read_boolean(input)?;
        GeoComplexPolygon::new(planet_model, points_list, test_point, in_set)
    }

    /// `isInSet(x, y, z, testPoint1, ...)`.
    fn is_in_set(&self, x: f64, y: f64, z: f64) -> Result<bool> {
        let pm = &*self.planet_model;
        let poly = &self.inner;
        let test_point = &self.test_point1;
        let test_point_in_set = self.test_point1_in_set;
        // If we're right on top of the point, we know the answer.
        if test_point.is_numerically_identical_xyz(x, y, z) {
            return Ok(test_point_in_set);
        }
        let parity = |it: &dyn CountingEdgeIterator, in_set: bool| -> bool {
            it.is_on_edge()
                || if (it.crossing_count() & 1) == 0 {
                    in_set
                } else {
                    !in_set
                }
        };
        // If we're right on top of any of the test planes, we navigate
        // solely on that plane: the XZ plane (fixed y), then the YZ plane
        // (fixed x), then the XY plane (fixed z).
        let fixed = [
            (
                &self.test_point1_fixed_y_plane,
                &self.test_point1_fixed_y_above_plane,
                &self.test_point1_fixed_y_below_plane,
                &poly.y_tree,
                test_point.y,
            ),
            (
                &self.test_point1_fixed_x_plane,
                &self.test_point1_fixed_x_above_plane,
                &self.test_point1_fixed_x_below_plane,
                &poly.x_tree,
                test_point.x,
            ),
            (
                &self.test_point1_fixed_z_plane,
                &self.test_point1_fixed_z_above_plane,
                &self.test_point1_fixed_z_below_plane,
                &poly.z_tree,
                test_point.z,
            ),
        ];
        for (plane, above, below, tree, value) in fixed {
            if let (Some(above), Some(below)) = (above, below) {
                if plane.evaluate_is_zero_xyz(x, y, z) {
                    let it = create_linear_crossing_edge_iterator(
                        poly, test_point, plane, above, below, x, y, z,
                    );
                    let mut it = it?;
                    tree.traverse_value(poly, &mut *it, value);
                    return Ok(parity(&*it, test_point_in_set));
                }
            }
        }
        // This is the expensive part!!
        // Changing the code below has an enormous impact on the queries per
        // second we see with the benchmark.
        // We need to use two planes to get there.  We don't know which two
        // planes will do it but we can figure it out.
        let travel_plane_fixed_x = Plane::new(1.0, 0.0, 0.0, -x);
        let travel_plane_fixed_y = Plane::new(0.0, 1.0, 0.0, -y);
        let travel_plane_fixed_z = Plane::new(0.0, 0.0, 1.0, -z);
        let fixed_y_above_plane = offset_plane(pm, &travel_plane_fixed_y, true, Axis::Y);
        let fixed_y_below_plane = offset_plane(pm, &travel_plane_fixed_y, false, Axis::Y);
        let fixed_x_above_plane = offset_plane(pm, &travel_plane_fixed_x, true, Axis::X);
        let fixed_x_below_plane = offset_plane(pm, &travel_plane_fixed_x, false, Axis::X);
        let fixed_z_above_plane = offset_plane(pm, &travel_plane_fixed_z, true, Axis::Z);
        let fixed_z_below_plane = offset_plane(pm, &travel_plane_fixed_z, false, Axis::Z);
        // Find the intersection points for each one of these and the
        // complementary test point planes.
        let mut traversal_strategies: Vec<TraversalStrategy<'_>> = Vec::with_capacity(12);
        let ixy = pm.inverse_xy_scaling_squared;
        let iz = pm.inverse_z_scaling_squared;
        let ts = |tpa: &Option<Plane>,
                  tpb: &Option<Plane>,
                  fa: &Option<Plane>,
                  fb: &Option<Plane>|
         -> Option<(Plane, Plane, Plane, Plane)> {
            match (tpa, tpb, fa, fb) {
                (Some(a), Some(b), Some(c), Some(d)) => Some((*a, *b, *c, *d)),
                _ => None,
            }
        };
        // The six combinations, in Java's order. Each: the test point's fixed
        // plane (and the coordinate along it), the travel plane, the
        // inverse-scaling factors of the two check terms, and the trees.
        let tp = test_point;
        #[allow(clippy::type_complexity)]
        let combos: [(
            Option<(Plane, Plane, Plane, Plane)>,
            f64,
            f64,
            Plane,
            Plane,
            f64,
            f64,
            &Tree,
            &Tree,
            u8,
        ); 6] = [
            (
                ts(
                    &self.test_point1_fixed_y_above_plane,
                    &self.test_point1_fixed_y_below_plane,
                    &fixed_x_above_plane,
                    &fixed_x_below_plane,
                ),
                ixy,
                ixy,
                self.test_point1_fixed_y_plane,
                travel_plane_fixed_x,
                tp.y,
                x,
                &poly.y_tree,
                &poly.x_tree,
                0,
            ),
            (
                ts(
                    &self.test_point1_fixed_z_above_plane,
                    &self.test_point1_fixed_z_below_plane,
                    &fixed_x_above_plane,
                    &fixed_x_below_plane,
                ),
                ixy,
                iz,
                self.test_point1_fixed_z_plane,
                travel_plane_fixed_x,
                tp.z,
                x,
                &poly.z_tree,
                &poly.x_tree,
                1,
            ),
            (
                ts(
                    &self.test_point1_fixed_x_above_plane,
                    &self.test_point1_fixed_x_below_plane,
                    &fixed_y_above_plane,
                    &fixed_y_below_plane,
                ),
                ixy,
                ixy,
                self.test_point1_fixed_x_plane,
                travel_plane_fixed_y,
                tp.x,
                y,
                &poly.x_tree,
                &poly.y_tree,
                2,
            ),
            (
                ts(
                    &self.test_point1_fixed_z_above_plane,
                    &self.test_point1_fixed_z_below_plane,
                    &fixed_y_above_plane,
                    &fixed_y_below_plane,
                ),
                iz,
                ixy,
                self.test_point1_fixed_z_plane,
                travel_plane_fixed_y,
                tp.z,
                y,
                &poly.z_tree,
                &poly.y_tree,
                3,
            ),
            (
                ts(
                    &self.test_point1_fixed_x_above_plane,
                    &self.test_point1_fixed_x_below_plane,
                    &fixed_z_above_plane,
                    &fixed_z_below_plane,
                ),
                ixy,
                iz,
                self.test_point1_fixed_x_plane,
                travel_plane_fixed_z,
                tp.x,
                z,
                &poly.x_tree,
                &poly.z_tree,
                4,
            ),
            (
                ts(
                    &self.test_point1_fixed_y_above_plane,
                    &self.test_point1_fixed_y_below_plane,
                    &fixed_z_above_plane,
                    &fixed_z_below_plane,
                ),
                ixy,
                iz,
                self.test_point1_fixed_y_plane,
                travel_plane_fixed_z,
                tp.y,
                z,
                &poly.y_tree,
                &poly.z_tree,
                5,
            ),
        ];
        for (
            planes,
            scale_a,
            scale_b,
            test_point_plane,
            travel_plane,
            first_leg_value,
            second_leg_value,
            first_tree,
            second_tree,
            which,
        ) in combos
        {
            let Some((tp_above, tp_below, f_above, f_below)) = planes else {
                continue;
            };
            // Java's check terms, operand order as written in each case.
            let (check_above, check_below) = match which {
                0 | 1 => (
                    4.0 * (f_above.D * f_above.D * scale_a + tp_above.D * tp_above.D * scale_b
                        - 1.0),
                    4.0 * (f_below.D * f_below.D * scale_a + tp_below.D * tp_below.D * scale_b
                        - 1.0),
                ),
                _ => (
                    4.0 * (tp_above.D * tp_above.D * scale_a + f_above.D * f_above.D * scale_b
                        - 1.0),
                    4.0 * (tp_below.D * tp_below.D * scale_a + f_below.D * f_below.D * scale_b
                        - 1.0),
                ),
            };
            if !(check_above < MINIMUM_RESOLUTION_SQUARED
                && check_below < MINIMUM_RESOLUTION_SQUARED)
            {
                continue;
            }
            let intersections = travel_plane
                .find_intersections(pm, &test_point_plane, &[])
                .unwrap_or_default();
            for p in intersections {
                // Java's per-case distance terms.
                let (tp_delta1, tp_delta2, cp_delta1, cp_delta2) = match which {
                    0 => (tp.x - p.x, tp.z - p.z, y - p.y, z - p.z),
                    1 => (tp.x - p.x, tp.y - p.y, y - p.y, z - p.z),
                    2 => (tp.y - p.y, tp.z - p.z, x - p.x, z - p.z),
                    3 => (tp.x - p.x, tp.y - p.y, x - p.x, z - p.z),
                    4 => (tp.y - p.y, tp.z - p.z, y - p.y, x - p.x),
                    _ => (tp.x - p.x, tp.z - p.z, y - p.y, x - p.x),
                };
                let new_distance = tp_delta1 * tp_delta1
                    + tp_delta2 * tp_delta2
                    + cp_delta1 * cp_delta1
                    + cp_delta2 * cp_delta2;
                traversal_strategies.push(TraversalStrategy {
                    traversal_distance: new_distance,
                    first_leg_value,
                    second_leg_value,
                    first_leg_plane: test_point_plane,
                    first_leg_above_plane: tp_above,
                    first_leg_below_plane: tp_below,
                    second_leg_plane: travel_plane,
                    second_leg_above_plane: f_above,
                    second_leg_below_plane: f_below,
                    first_leg_tree: first_tree,
                    second_leg_tree: second_tree,
                    intersection_point: p,
                });
            }
        }
        traversal_strategies.sort_by(|a, b| {
            if a.traversal_distance < b.traversal_distance {
                std::cmp::Ordering::Less
            } else if a.traversal_distance > b.traversal_distance {
                std::cmp::Ordering::Greater
            } else {
                std::cmp::Ordering::Equal
            }
        });
        if traversal_strategies.is_empty() {
            return Err(illegal("No dual-plane travel strategies were found"));
        }
        // Loop through travel strategies, in order, until we find one that
        // works.
        for ts in &traversal_strategies {
            match ts.apply(poly, test_point, test_point_in_set, x, y, z) {
                Ok(v) => return Ok(v),
                Err(Error::IllegalArgument(_)) => {
                    // Continue
                }
                Err(e) => return Err(e),
            }
        }
        Err(illegal("Exhausted all traversal strategies"))
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let mut minimum_distance = f64::INFINITY;
        for &shape_start_edge in &self.shape_start_edges {
            let mut shape_edge = shape_start_edge;
            loop {
                let e = &self.inner.edges[shape_edge];
                let new_dist = style.compute_distance(&e.start_point, x, y, z);
                if new_dist < minimum_distance {
                    minimum_distance = new_dist;
                }
                let new_plane_dist = style.compute_distance_to_plane(
                    &self.planet_model,
                    &e.plane,
                    x,
                    y,
                    z,
                    &[&e.start_plane, &e.end_plane],
                );
                if new_plane_dist < minimum_distance {
                    minimum_distance = new_plane_dist;
                }
                shape_edge = e.next;
                if shape_edge == shape_start_edge {
                    break;
                }
            }
        }
        minimum_distance
    }

    /// The narrowest axis's tree traversal shared by both `intersects`.
    fn traverse_narrowest(&self, it: &mut dyn EdgeIterator, b: &XYZBounds) -> bool {
        let poly = &self.inner;
        let nan = f64::NAN;
        let (min_x, max_x) = (b.minimum_x().unwrap_or(nan), b.maximum_x().unwrap_or(nan));
        let (min_y, max_y) = (b.minimum_y().unwrap_or(nan), b.maximum_y().unwrap_or(nan));
        let (min_z, max_z) = (b.minimum_z().unwrap_or(nan), b.maximum_z().unwrap_or(nan));
        let x_delta = max_x - min_x;
        let y_delta = max_y - min_y;
        let z_delta = max_z - min_z;
        if x_delta <= y_delta && x_delta <= z_delta {
            return !poly.x_tree.traverse(poly, it, min_x, max_x);
        } else if y_delta <= x_delta && y_delta <= z_delta {
            return !poly.y_tree.traverse(poly, it, min_y, max_y);
        } else if z_delta <= x_delta && z_delta <= y_delta {
            return !poly.z_tree.traverse(poly, it, min_z, max_z);
        }
        true
    }
}

impl SerializableObject for GeoComplexPolygon {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_int(out, self.points_list.len() as i32);
        for points in &self.points_list {
            write_point_array(out, points);
        }
        self.test_point1.write(out);
        write_boolean(out, self.test_point1_in_set);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(6)
    }
}

impl_planet_object!(GeoComplexPolygon);
impl_membership_shape!(GeoComplexPolygon);
impl_base_area!(GeoComplexPolygon);

impl Membership for GeoComplexPolygon {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        match self.is_in_set(x, y, z) {
            Ok(v) => v,
            Err(e) => {
                raise(e);
                false
            }
        }
    }
}

impl Bounded for GeoComplexPolygon {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        base_get_bounds(self, &self.planet_model, bounds);
        for &start_edge in &self.shape_start_edges {
            let mut current_edge = start_edge;
            loop {
                let e = &self.inner.edges[current_edge];
                bounds.add_point(&e.start_point);
                bounds.add_plane(
                    &self.planet_model,
                    &e.plane,
                    &[&e.start_plane, &e.end_plane],
                );
                current_edge = e.next;
                if current_edge == start_edge {
                    break;
                }
            }
        }
    }
}

impl GeoShape for GeoComplexPolygon {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        // Create the intersector
        let mut intersector = IntersectorEdgeIterator {
            plane: p,
            notable_points,
            bounds,
        };
        // First, compute the bounds for the the plane
        let mut xyz_bounds = XYZBounds::new();
        p.record_bounds_xyz(&self.planet_model, &mut xyz_bounds, bounds);
        for point in notable_points {
            xyz_bounds.add_point(point);
        }
        // If we have no bounds at all then the answer is "false"
        if xyz_bounds.maximum_x().is_none()
            || xyz_bounds.minimum_x().is_none()
            || xyz_bounds.maximum_y().is_none()
            || xyz_bounds.minimum_y().is_none()
            || xyz_bounds.maximum_z().is_none()
            || xyz_bounds.minimum_z().is_none()
        {
            return false;
        }
        // Figure out which tree likely works best
        self.traverse_narrowest(&mut intersector, &xyz_bounds)
    }
}

impl GeoAreaShape for GeoComplexPolygon {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        // Create the intersector
        let mut intersector = IntersectorShapeIterator { shape: geo_shape };
        // First, compute the bounds for the the plane
        let mut xyz_bounds = XYZBounds::new();
        geo_shape.get_bounds(&mut xyz_bounds);
        // Figure out which tree likely works best
        self.traverse_narrowest(&mut intersector, &xyz_bounds)
    }
}

impl GeoPolygon for GeoComplexPolygon {}

/// `TraversalStrategy`: a two-leg path from the test point to the point.
struct TraversalStrategy<'a> {
    traversal_distance: f64,
    first_leg_value: f64,
    second_leg_value: f64,
    first_leg_plane: Plane,
    first_leg_above_plane: Plane,
    first_leg_below_plane: Plane,
    second_leg_plane: Plane,
    second_leg_above_plane: Plane,
    second_leg_below_plane: Plane,
    first_leg_tree: &'a Tree,
    second_leg_tree: &'a Tree,
    intersection_point: GeoPoint,
}

impl TraversalStrategy<'_> {
    fn apply(
        &self,
        poly: &Inner,
        test_point: &GeoPoint,
        test_point_in_set: bool,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<bool> {
        let ip = &self.intersection_point;
        // First, try with two individual legs.  If that doesn't work, try
        // the DualCrossingIterator.
        let first_try = catch(|| -> Result<bool> {
            // First, we'll determine if the intersection point is in set or
            // not
            let test_point_edge_iterator = create_linear_crossing_edge_iterator(
                poly,
                test_point,
                &self.first_leg_plane,
                &self.first_leg_above_plane,
                &self.first_leg_below_plane,
                ip.x,
                ip.y,
                ip.z,
            );
            let mut test_point_edge_iterator = test_point_edge_iterator?;
            // Traverse our way from the test point to the check point.  Use
            // the z tree because that's fixed.
            self.first_leg_tree.traverse_value(
                poly,
                &mut *test_point_edge_iterator,
                self.first_leg_value,
            );
            let intersection_point_on_edge = test_point_edge_iterator.is_on_edge();
            // If the intersection point is on the edge, we cannot use this
            // combination of legs, since it's not logically possible to
            // compute in-set or out-of-set for such a point.
            if intersection_point_on_edge {
                return Err(illegal(
                    "Intersection point landed on an edge -- illegal path",
                ));
            }
            let intersection_point_in_set = intersection_point_on_edge
                || if (test_point_edge_iterator.crossing_count() & 1) == 0 {
                    test_point_in_set
                } else {
                    !test_point_in_set
                };
            // Now do the final leg
            let travel_edge_iterator = create_linear_crossing_edge_iterator(
                poly,
                ip,
                &self.second_leg_plane,
                &self.second_leg_above_plane,
                &self.second_leg_below_plane,
                x,
                y,
                z,
            );
            let mut travel_edge_iterator = travel_edge_iterator?;
            // Traverse our way from the test point to the check point.
            self.second_leg_tree.traverse_value(
                poly,
                &mut *travel_edge_iterator,
                self.second_leg_value,
            );
            Ok(travel_edge_iterator.is_on_edge()
                || if (travel_edge_iterator.crossing_count() & 1) == 0 {
                    intersection_point_in_set
                } else {
                    !intersection_point_in_set
                })
        })
        .and_then(|r| r);
        match first_try {
            Ok(v) => Ok(v),
            Err(Error::IllegalArgument(_)) => {
                // Try other iterator
                catch(|| -> Result<bool> {
                    let edge_iterator = DualCrossingEdgeIterator::new(
                        poly,
                        test_point,
                        &self.first_leg_plane,
                        &self.first_leg_above_plane,
                        &self.first_leg_below_plane,
                        &self.second_leg_plane,
                        &self.second_leg_above_plane,
                        &self.second_leg_below_plane,
                        x,
                        y,
                        z,
                        ip.clone(),
                    );
                    let mut edge_iterator = edge_iterator?;
                    // Compute the crossings
                    self.first_leg_tree.traverse_value(
                        poly,
                        &mut edge_iterator,
                        self.first_leg_value,
                    );
                    if edge_iterator.is_on_edge() {
                        return Ok(true);
                    }
                    self.second_leg_tree.traverse_value(
                        poly,
                        &mut edge_iterator,
                        self.second_leg_value,
                    );
                    Ok(edge_iterator.is_on_edge()
                        || if (edge_iterator.crossing_count() & 1) == 0 {
                            test_point_in_set
                        } else {
                            !test_point_in_set
                        })
                })
                .and_then(|r| r)
            }
            Err(e) => Err(e),
        }
    }
}

/// `IntersectorEdgeIterator`: stops at an edge the plane meets.
struct IntersectorEdgeIterator<'a> {
    plane: &'a Plane,
    notable_points: &'a [GeoPoint],
    bounds: &'a [&'a dyn Membership],
}

impl EdgeIterator for IntersectorEdgeIterator<'_> {
    fn matches(&mut self, poly: &Inner, edge: usize) -> bool {
        let e = &poly.edges[edge];
        !self.plane.intersects(
            &poly.planet_model,
            &e.plane,
            self.notable_points,
            &e.notable_points,
            self.bounds,
            &[&e.start_plane, &e.end_plane],
        )
    }
}

/// `IntersectorShapeIterator`: stops at an edge the shape meets.
struct IntersectorShapeIterator<'a> {
    shape: &'a dyn GeoShape,
}

impl EdgeIterator for IntersectorShapeIterator<'_> {
    fn matches(&mut self, poly: &Inner, edge: usize) -> bool {
        let e = &poly.edges[edge];
        !self
            .shape
            .intersects(&e.plane, &e.notable_points, &[&e.start_plane, &e.end_plane])
    }
}

/// `createLinearCrossingEdgeIterator(...)`: the sector iterator, or the full
/// one where the sector's bound planes cannot be built.
#[allow(clippy::too_many_arguments)]
fn create_linear_crossing_edge_iterator<'a>(
    poly: &Inner,
    test_point: &GeoPoint,
    plane: &'a Plane,
    above_plane: &'a Plane,
    below_plane: &'a Plane,
    the_point_x: f64,
    the_point_y: f64,
    the_point_z: f64,
) -> Result<Box<dyn CountingEdgeIterator + 'a>> {
    let _ = poly;
    match SectorLinearCrossingEdgeIterator::new(
        test_point,
        plane,
        above_plane,
        below_plane,
        the_point_x,
        the_point_y,
        the_point_z,
    ) {
        Ok(it) => Ok(Box::new(it)),
        Err(Error::IllegalArgument(_)) => Ok(Box::new(FullLinearCrossingEdgeIterator::new(
            test_point,
            plane,
            above_plane,
            below_plane,
            the_point_x,
            the_point_y,
            the_point_z,
        )?)),
        Err(e) => Err(e),
    }
}

/// `findAdjoiningPoints(plane, pointOnPlane, envelopePlane)`: two surface
/// points either side of the plane, off the envelope plane; `None` when
/// none are found in `MAX_ITERATIONS` steps. A degenerate perpendicular
/// throws in Java (raised here).
fn find_adjoining_points(
    pm: &PlanetModel,
    plane: &Plane,
    point_on_plane: &GeoPoint,
    envelope_plane: &Plane,
) -> Option<[GeoPoint; 2]> {
    // Compute a normalized perpendicular vector
    let perpendicular = match Vector::perpendicular(plane, point_on_plane) {
        Ok(v) => v,
        Err(e) => {
            raise(e);
            return None;
        }
    };
    let mut distance_factor = 0.0;
    for _ in 0..MAX_ITERATIONS {
        distance_factor += DELTA_DISTANCE;
        // Compute two new points along this vector from the original
        let point_a = pm.create_surface_point_xyz(
            point_on_plane.x + perpendicular.x * distance_factor,
            point_on_plane.y + perpendicular.y * distance_factor,
            point_on_plane.z + perpendicular.z * distance_factor,
        );
        let point_b = pm.create_surface_point_xyz(
            point_on_plane.x - perpendicular.x * distance_factor,
            point_on_plane.y - perpendicular.y * distance_factor,
            point_on_plane.z - perpendicular.z * distance_factor,
        );
        if abs(envelope_plane.evaluate(&point_a)) > OFF_PLANE_AMOUNT
            && abs(envelope_plane.evaluate(&point_b)) > OFF_PLANE_AMOUNT
        {
            return Some([point_a, point_b]);
        }
    }
    None
}

/// The crossings an edge makes with an envelope plane within the bounds
/// (each confirmed by `edge_crosses_envelope`).
fn count_crossings_with(
    poly: &Inner,
    edge: &Edge,
    envelope_plane: &Plane,
    envelope_bounds: &[&dyn Membership],
    crosses: impl Fn(&GeoPoint) -> bool,
) -> i32 {
    let intersections =
        edge.plane
            .find_intersections_two(&poly.planet_model, envelope_plane, envelope_bounds);
    let mut crossings = 0;
    {
        for intersection in intersections.iter().flatten() {
            if edge.start_plane.strictly_within(intersection)
                && edge.end_plane.strictly_within(intersection)
            {
                crossings += i32::from(crosses(intersection));
            }
        }
    }
    crossings
}

/// `FullLinearCrossingEdgeIterator`.
struct FullLinearCrossingEdgeIterator<'a> {
    plane: &'a Plane,
    above_plane: &'a Plane,
    below_plane: &'a Plane,
    bound: SidedPlane,
    the_point_x: f64,
    the_point_y: f64,
    the_point_z: f64,
    on_edge: bool,
    above_crossing_count: i32,
    below_crossing_count: i32,
}

impl<'a> FullLinearCrossingEdgeIterator<'a> {
    fn new(
        test_point: &GeoPoint,
        plane: &'a Plane,
        above_plane: &'a Plane,
        below_plane: &'a Plane,
        the_point_x: f64,
        the_point_y: f64,
        the_point_z: f64,
    ) -> Result<Self> {
        if plane.v.is_numerically_identical(test_point) {
            return Err(illegal("Plane vector identical to testpoint vector"));
        }
        // It doesn't matter which 1/2 of the world we choose, but we must
        // choose only one.
        let bound = SidedPlane::from_two_vectors(&plane.v, test_point)?;
        Ok(FullLinearCrossingEdgeIterator {
            plane,
            above_plane,
            below_plane,
            bound,
            the_point_x,
            the_point_y,
            the_point_z,
            on_edge: false,
            above_crossing_count: 0,
            below_crossing_count: 0,
        })
    }

    fn edge_crosses_envelope(
        &self,
        pm: &PlanetModel,
        edge_plane: &Plane,
        intersection_point: &GeoPoint,
        envelope_plane: &Plane,
    ) -> bool {
        let Some(adjoining_points) =
            find_adjoining_points(pm, edge_plane, intersection_point, envelope_plane)
        else {
            return true;
        };
        let mut within_count = 0;
        for adjoining in &adjoining_points {
            if self.plane.evaluate_is_zero(adjoining) && self.bound.is_within(adjoining) {
                within_count += 1;
            }
        }
        (within_count & 1) != 0
    }
}

impl EdgeIterator for FullLinearCrossingEdgeIterator<'_> {
    fn matches(&mut self, poly: &Inner, edge_index: usize) -> bool {
        let edge = &poly.edges[edge_index];
        let pm = &*poly.planet_model;
        // Early exit if the point is on the edge.
        if edge.is_within(self.the_point_x, self.the_point_y, self.the_point_z) {
            self.on_edge = true;
            return false;
        }
        // This should precisely mirror what is in DualCrossingIterator, but
        // without the dual crossings.
        let plane_crossings = self.plane.find_intersections_arr(
            pm,
            &edge.plane,
            &[&self.bound, &edge.start_plane, &edge.end_plane],
        );
        if let Some(c) = &plane_crossings {
            if c[0].is_none()
                && !self.plane.evaluate_is_zero(&edge.start_point)
                && !self.plane.evaluate_is_zero(&edge.end_point)
            {
                return true;
            }
        }
        // Determine crossings of this edge against all inside/outside planes.
        let above_crossings =
            count_crossings_with(poly, edge, self.above_plane, &[&self.bound], |p| {
                self.edge_crosses_envelope(pm, &edge.plane, p, self.above_plane)
            });
        self.above_crossing_count += above_crossings;
        let below_crossings =
            count_crossings_with(poly, edge, self.below_plane, &[&self.bound], |p| {
                self.edge_crosses_envelope(pm, &edge.plane, p, self.below_plane)
            });
        self.below_crossing_count += below_crossings;
        true
    }
}

impl CountingEdgeIterator for FullLinearCrossingEdgeIterator<'_> {
    fn crossing_count(&self) -> i32 {
        self.above_crossing_count.min(self.below_crossing_count)
    }

    fn is_on_edge(&self) -> bool {
        self.on_edge
    }
}

/// `SectorLinearCrossingEdgeIterator`.
struct SectorLinearCrossingEdgeIterator<'a> {
    plane: &'a Plane,
    above_plane: &'a Plane,
    below_plane: &'a Plane,
    bound1: SidedPlane,
    bound2: SidedPlane,
    the_point_x: f64,
    the_point_y: f64,
    the_point_z: f64,
    on_edge: bool,
    above_crossing_count: i32,
    below_crossing_count: i32,
}

impl<'a> SectorLinearCrossingEdgeIterator<'a> {
    fn new(
        test_point: &GeoPoint,
        plane: &'a Plane,
        above_plane: &'a Plane,
        below_plane: &'a Plane,
        the_point_x: f64,
        the_point_y: f64,
        the_point_z: f64,
    ) -> Result<Self> {
        // We have to be sure we don't accidently create two bounds that
        // would exclude all points. Not sure about how to do this yet...
        let bound1_plane = SidedPlane::from_xyz_vectors(
            the_point_x,
            the_point_y,
            the_point_z,
            &plane.v,
            test_point,
        );
        let bound1_plane = bound1_plane?;
        let bound2_plane = SidedPlane::from_vector_xyz(
            test_point,
            &plane.v,
            the_point_x,
            the_point_y,
            the_point_z,
        );
        let bound2_plane = bound2_plane?;
        if bound1_plane.is_numerically_identical_plane(&bound2_plane) {
            return Err(illegal(
                "Sector iterator unreliable when bounds planes are numerically identical",
            ));
        }
        Ok(SectorLinearCrossingEdgeIterator {
            plane,
            above_plane,
            below_plane,
            bound1: bound1_plane,
            bound2: bound2_plane,
            the_point_x,
            the_point_y,
            the_point_z,
            on_edge: false,
            above_crossing_count: 0,
            below_crossing_count: 0,
        })
    }

    fn edge_crosses_envelope(
        &self,
        pm: &PlanetModel,
        edge_plane: &Plane,
        intersection_point: &GeoPoint,
        envelope_plane: &Plane,
    ) -> bool {
        let Some(adjoining_points) =
            find_adjoining_points(pm, edge_plane, intersection_point, envelope_plane)
        else {
            return true;
        };
        let mut within_count = 0;
        for adjoining in &adjoining_points {
            if self.plane.evaluate_is_zero(adjoining)
                && self.bound1.is_within(adjoining)
                && self.bound2.is_within(adjoining)
            {
                within_count += 1;
            }
        }
        (within_count & 1) != 0
    }
}

impl EdgeIterator for SectorLinearCrossingEdgeIterator<'_> {
    fn matches(&mut self, poly: &Inner, edge_index: usize) -> bool {
        let edge = &poly.edges[edge_index];
        let pm = &*poly.planet_model;
        // Early exit if the point is on the edge.
        if edge.is_within(self.the_point_x, self.the_point_y, self.the_point_z) {
            self.on_edge = true;
            return false;
        }
        // This should precisely mirror what is in DualCrossingIterator, but
        // without the dual crossings.
        let plane_crossings = self.plane.find_intersections_arr(
            pm,
            &edge.plane,
            &[
                &self.bound1,
                &self.bound2,
                &edge.start_plane,
                &edge.end_plane,
            ],
        );
        if let Some(c) = &plane_crossings {
            if c[0].is_none()
                && !self.plane.evaluate_is_zero(&edge.start_point)
                && !self.plane.evaluate_is_zero(&edge.end_point)
            {
                return true;
            }
        }
        // Determine crossings of this edge against all inside/outside planes.
        let above_crossings = count_crossings_with(
            poly,
            edge,
            self.above_plane,
            &[&self.bound1, &self.bound2],
            |p| self.edge_crosses_envelope(pm, &edge.plane, p, self.above_plane),
        );
        self.above_crossing_count += above_crossings;
        let below_crossings = count_crossings_with(
            poly,
            edge,
            self.below_plane,
            &[&self.bound1, &self.bound2],
            |p| self.edge_crosses_envelope(pm, &edge.plane, p, self.below_plane),
        );
        self.below_crossing_count += below_crossings;
        true
    }
}

impl CountingEdgeIterator for SectorLinearCrossingEdgeIterator<'_> {
    fn crossing_count(&self) -> i32 {
        self.above_crossing_count.min(self.below_crossing_count)
    }

    fn is_on_edge(&self) -> bool {
        self.on_edge
    }
}

/// The inside/outside planes `computeInsideOutside` chooses.
struct InsideOutside {
    test_point_inside_plane: Plane,
    test_point_outside_plane: Plane,
    travel_inside_plane: Plane,
    travel_outside_plane: Plane,
    inside_test_point_cutoff_plane: SidedPlane,
    inside_travel_cutoff_plane: SidedPlane,
    outside_test_point_cutoff_plane: SidedPlane,
    outside_travel_cutoff_plane: SidedPlane,
}

/// `DualCrossingEdgeIterator`.
struct DualCrossingEdgeIterator<'a> {
    seen_edges: Vec<bool>,
    test_point: &'a GeoPoint,
    test_point_plane: &'a Plane,
    test_point_above_plane: &'a Plane,
    test_point_below_plane: &'a Plane,
    travel_plane: &'a Plane,
    travel_above_plane: &'a Plane,
    travel_below_plane: &'a Plane,
    the_point_x: f64,
    the_point_y: f64,
    the_point_z: f64,
    intersection_point: GeoPoint,
    test_point_cutoff_plane: SidedPlane,
    check_point_cutoff_plane: SidedPlane,
    test_point_other_cutoff_plane: SidedPlane,
    check_point_other_cutoff_plane: SidedPlane,
    inside_outside: Option<InsideOutside>,
    on_edge: bool,
    inner_crossing_count: i32,
    outer_crossing_count: i32,
}

impl<'a> DualCrossingEdgeIterator<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        poly: &Inner,
        test_point: &'a GeoPoint,
        test_point_plane: &'a Plane,
        test_point_above_plane: &'a Plane,
        test_point_below_plane: &'a Plane,
        travel_plane: &'a Plane,
        travel_above_plane: &'a Plane,
        travel_below_plane: &'a Plane,
        the_point_x: f64,
        the_point_y: f64,
        the_point_z: f64,
        intersection_point: GeoPoint,
    ) -> Result<Self> {
        // We construct cutoff planes for the test point and the check point.
        let test_point_bound1 =
            SidedPlane::from_vectors(&intersection_point, &test_point_plane.v, test_point)?;
        let test_point_bound2 =
            SidedPlane::from_vectors(test_point, &test_point_plane.v, &intersection_point)?;
        if test_point_bound1.is_functionally_identical(&test_point_bound2) {
            return Err(illegal(
                "Dual iterator unreliable when bounds planes are functionally identical",
            ));
        }
        let check_point_bound1 = SidedPlane::from_vector_xyz(
            &intersection_point,
            &travel_plane.v,
            the_point_x,
            the_point_y,
            the_point_z,
        );
        let check_point_bound1 = check_point_bound1?;
        let check_point_bound2 = SidedPlane::from_xyz_vectors(
            the_point_x,
            the_point_y,
            the_point_z,
            &travel_plane.v,
            &intersection_point,
        );
        let check_point_bound2 = check_point_bound2?;
        if check_point_bound1.is_functionally_identical(&check_point_bound2) {
            return Err(illegal(
                "Dual iterator unreliable when bounds planes are functionally identical",
            ));
        }
        Ok(DualCrossingEdgeIterator {
            seen_edges: vec![false; poly.edges.len()],
            test_point,
            test_point_plane,
            test_point_above_plane,
            test_point_below_plane,
            travel_plane,
            travel_above_plane,
            travel_below_plane,
            the_point_x,
            the_point_y,
            the_point_z,
            intersection_point,
            test_point_cutoff_plane: test_point_bound1,
            test_point_other_cutoff_plane: test_point_bound2,
            check_point_cutoff_plane: check_point_bound1,
            check_point_other_cutoff_plane: check_point_bound2,
            inside_outside: None,
            on_edge: false,
            inner_crossing_count: 0,
            outer_crossing_count: 0,
        })
    }

    /// `computeInsideOutside()`: once, on first use.
    fn compute_inside_outside(&mut self, pm: &PlanetModel) -> Result<()> {
        if self.inside_outside.is_some() {
            return Ok(());
        }
        // Convert travel plane to a sided plane
        let intersection_bound1 =
            SidedPlane::from_normal(self.test_point, &self.travel_plane.v, self.travel_plane.D)?;
        // Convert the test point plane to a sided plane
        let intersection_bound2 = SidedPlane::from_xyz_normal(
            self.the_point_x,
            self.the_point_y,
            self.the_point_z,
            &self.test_point_plane.v,
            self.test_point_plane.D,
        );
        let intersection_bound2 = intersection_bound2?;
        let b: [&dyn Membership; 2] = [&intersection_bound1, &intersection_bound2];
        let find = |p: &Plane, q: &Plane| p.find_intersections(pm, q, &b).unwrap_or_default();
        let above_above = find(self.travel_above_plane, self.test_point_above_plane);
        let above_below = find(self.travel_above_plane, self.test_point_below_plane);
        let below_below = find(self.travel_below_plane, self.test_point_below_plane);
        let below_above = find(self.travel_below_plane, self.test_point_above_plane);
        let (
            travel_inside_plane,
            test_point_inside_plane,
            travel_outside_plane,
            test_point_outside_plane,
            inside_inside_points,
        ) = if !above_above.is_empty() {
            (
                self.travel_above_plane,
                self.test_point_above_plane,
                self.travel_below_plane,
                self.test_point_below_plane,
                above_above,
            )
        } else if !above_below.is_empty() {
            (
                self.travel_above_plane,
                self.test_point_below_plane,
                self.travel_below_plane,
                self.test_point_above_plane,
                above_below,
            )
        } else if !below_below.is_empty() {
            (
                self.travel_below_plane,
                self.test_point_below_plane,
                self.travel_above_plane,
                self.test_point_above_plane,
                below_below,
            )
        } else if !below_above.is_empty() {
            (
                self.travel_below_plane,
                self.test_point_above_plane,
                self.travel_above_plane,
                self.test_point_below_plane,
                below_above,
            )
        } else {
            return Err(Error::IllegalState(format!(
                "Can't find traversal intersection among: {}, {}, {}, {}",
                self.travel_above_plane,
                self.test_point_above_plane,
                self.travel_below_plane,
                self.test_point_below_plane
            )));
        };
        // Get the inside-inside intersection point.  Note that we don't
        // bother checking to see if the intersection point is within these
        // planes; we just pick the one closest to the intersection point.
        let inside_inside_point = self.pick_proximate(&inside_inside_points)?;
        // Get the outside-outside intersection point
        let outside_outside_points = test_point_outside_plane
            .find_intersections(pm, travel_outside_plane, &[])
            .unwrap_or_default();
        let outside_outside_point = self.pick_proximate(&outside_outside_points)?;
        let (tx, ty, tz) = (self.the_point_x, self.the_point_y, self.the_point_z);
        let io = InsideOutside {
            inside_travel_cutoff_plane: SidedPlane::from_xyz_vectors(
                tx,
                ty,
                tz,
                &travel_inside_plane.v,
                &inside_inside_point,
            )?,
            outside_travel_cutoff_plane: SidedPlane::from_xyz_vectors(
                tx,
                ty,
                tz,
                &travel_inside_plane.v,
                &outside_outside_point,
            )?,
            inside_test_point_cutoff_plane: SidedPlane::from_vectors(
                self.test_point,
                &test_point_inside_plane.v,
                &inside_inside_point,
            )?,
            outside_test_point_cutoff_plane: SidedPlane::from_vectors(
                self.test_point,
                &test_point_outside_plane.v,
                &outside_outside_point,
            )?,
            test_point_inside_plane: *test_point_inside_plane,
            test_point_outside_plane: *test_point_outside_plane,
            travel_inside_plane: *travel_inside_plane,
            travel_outside_plane: *travel_outside_plane,
        };
        self.inside_outside = Some(io);
        Ok(())
    }

    fn pick_proximate(&self, points: &[GeoPoint]) -> Result<GeoPoint> {
        match points.len() {
            0 => Err(illegal(
                "No off-plane intersection points were found; can't compute traversal",
            )),
            1 => Ok(points[0].clone()),
            _ => {
                let p1dist = compute_squared_distance(&points[0], &self.intersection_point);
                let p2dist = compute_squared_distance(&points[1], &self.intersection_point);
                if p1dist < p2dist {
                    Ok(points[0].clone())
                } else if p2dist < p1dist {
                    Ok(points[1].clone())
                } else {
                    Err(Error::IllegalArgument(format!(
                        "Neither off-plane intersection point matched intersection point; intersection = {}; offplane choice 0: {}; offplane choice 1: {}",
                        self.intersection_point, points[0], points[1]
                    )))
                }
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn count_crossings(
        &self,
        poly: &Inner,
        edge: &Edge,
        travel_envelope_plane: &Plane,
        travel_envelope_bound1: &dyn Membership,
        travel_envelope_bound2: &dyn Membership,
        test_point_envelope_plane: &Plane,
        test_point_envelope_bound1: &dyn Membership,
        test_point_envelope_bound2: &dyn Membership,
    ) -> i32 {
        let pm = &*poly.planet_model;
        let travel_intersections = edge.plane.find_intersections(
            pm,
            travel_envelope_plane,
            &[travel_envelope_bound1, travel_envelope_bound2],
        );
        let test_point_intersections = edge.plane.find_intersections(
            pm,
            test_point_envelope_plane,
            &[test_point_envelope_bound1, test_point_envelope_bound2],
        );
        let mut crossings = 0;
        if let Some(travel_intersections) = &travel_intersections {
            for intersection in travel_intersections {
                if edge.start_plane.strictly_within(intersection)
                    && edge.end_plane.strictly_within(intersection)
                {
                    // Make sure it's not a dup
                    let mut not_dup = true;
                    if let Some(tpi) = &test_point_intersections {
                        for other_intersection in tpi {
                            if edge.start_plane.strictly_within(other_intersection)
                                && edge.end_plane.strictly_within(other_intersection)
                                && intersection.is_numerically_identical(other_intersection)
                            {
                                not_dup = false;
                                break;
                            }
                        }
                    }
                    if !not_dup {
                        continue;
                    }
                    crossings += i32::from(self.edge_crosses_envelope(
                        pm,
                        &edge.plane,
                        intersection,
                        travel_envelope_plane,
                    ));
                }
            }
        }
        if let Some(tpi) = &test_point_intersections {
            for intersection in tpi {
                if edge.start_plane.strictly_within(intersection)
                    && edge.end_plane.strictly_within(intersection)
                {
                    crossings += i32::from(self.edge_crosses_envelope(
                        pm,
                        &edge.plane,
                        intersection,
                        test_point_envelope_plane,
                    ));
                }
            }
        }
        crossings
    }

    fn edge_crosses_envelope(
        &self,
        pm: &PlanetModel,
        edge_plane: &Plane,
        intersection_point: &GeoPoint,
        envelope_plane: &Plane,
    ) -> bool {
        let Some(adjoining_points) =
            find_adjoining_points(pm, edge_plane, intersection_point, envelope_plane)
        else {
            return true;
        };
        let mut within_count = 0;
        for adjoining in &adjoining_points {
            if (self.travel_plane.evaluate_is_zero(adjoining)
                && self.check_point_cutoff_plane.is_within(adjoining)
                && self.check_point_other_cutoff_plane.is_within(adjoining))
                || (self.test_point_plane.evaluate_is_zero(adjoining)
                    && self.test_point_cutoff_plane.is_within(adjoining)
                    && self.test_point_other_cutoff_plane.is_within(adjoining))
            {
                within_count += 1;
            }
        }
        (within_count & 1) != 0
    }
}

impl EdgeIterator for DualCrossingEdgeIterator<'_> {
    fn matches(&mut self, poly: &Inner, edge_index: usize) -> bool {
        let edge = &poly.edges[edge_index];
        let pm = &*poly.planet_model;
        // Early exit if the point is on the edge, in which case we
        // accidentally discovered the answer.
        if edge.is_within(self.the_point_x, self.the_point_y, self.the_point_z) {
            self.on_edge = true;
            return false;
        }
        // All edges that touch the travel planes get assessed the same.  So,
        // for each intersecting edge on both legs: (1) If the edge contains
        // the intersection point, we analyze it on only one leg.  For the
        // other leg, we do nothing.
        if self.seen_edges[edge_index] {
            return true;
        }
        self.seen_edges[edge_index] = true;
        // We've never seen this edge before.  Evaluate it in the context of
        // inner and outer planes.
        if let Err(e) = self.compute_inside_outside(pm) {
            raise(e);
            return true;
        }
        let travel_crossings = self.travel_plane.find_intersections(
            pm,
            &edge.plane,
            &[
                &self.check_point_cutoff_plane,
                &self.check_point_other_cutoff_plane,
                &edge.start_plane,
                &edge.end_plane,
            ],
        );
        if let Some(tc) = &travel_crossings {
            if tc.is_empty() {
                // Check test point plane
                let test_point_crossings = self.test_point_plane.find_intersections(
                    pm,
                    &edge.plane,
                    &[
                        &self.test_point_cutoff_plane,
                        &self.test_point_other_cutoff_plane,
                        &edge.start_plane,
                        &edge.end_plane,
                    ],
                );
                if let Some(tpc) = &test_point_crossings {
                    if tpc.is_empty()
                        && !self.travel_plane.evaluate_is_zero(&edge.start_point)
                        && !self.travel_plane.evaluate_is_zero(&edge.end_point)
                        && !self.test_point_plane.evaluate_is_zero(&edge.start_point)
                        && !self.test_point_plane.evaluate_is_zero(&edge.end_point)
                    {
                        return true;
                    }
                }
            }
        }
        // Determine crossings of this edge against all inside/outside planes.
        let io = self.inside_outside.as_ref().expect("computed above");
        let inner = self.count_crossings(
            poly,
            edge,
            &io.travel_inside_plane,
            &self.check_point_cutoff_plane,
            &io.inside_travel_cutoff_plane,
            &io.test_point_inside_plane,
            &self.test_point_cutoff_plane,
            &io.inside_test_point_cutoff_plane,
        );
        let outer = self.count_crossings(
            poly,
            edge,
            &io.travel_outside_plane,
            &self.check_point_cutoff_plane,
            &io.outside_travel_cutoff_plane,
            &io.test_point_outside_plane,
            &self.test_point_cutoff_plane,
            &io.outside_test_point_cutoff_plane,
        );
        self.inner_crossing_count += inner;
        self.outer_crossing_count += outer;
        true
    }
}

impl CountingEdgeIterator for DualCrossingEdgeIterator<'_> {
    fn crossing_count(&self) -> i32 {
        // Doesn't return the actual crossing count -- just gets the
        // even/odd part right
        self.inner_crossing_count.min(self.outer_crossing_count)
    }

    fn is_on_edge(&self) -> bool {
        self.on_edge
    }
}

/// `computeSquaredDistance(checkPoint, intersectionPoint)`.
fn compute_squared_distance(check_point: &GeoPoint, intersection_point: &GeoPoint) -> f64 {
    let distance_x = check_point.x - intersection_point.x;
    let distance_y = check_point.y - intersection_point.y;
    let distance_z = check_point.z - intersection_point.z;
    distance_x * distance_x + distance_y * distance_y + distance_z * distance_z
}
