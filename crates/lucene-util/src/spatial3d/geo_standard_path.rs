//! `GeoStandardPath` (`org.apache.lucene.spatial3d.geom.GeoStandardPath`): a
//! path of great-circle segments with a width, as a balanced tree of
//! components -- segments (four planes) and the end caps between them
//! (circle planes cut off at the segments' ends).
//!
//! Java's components reference each other (`previous`) and form a tree of
//! objects; here segments and end points live in two vectors on the path and
//! the tree and `previous` hold indexes into them.

#![allow(non_snake_case)]

use std::sync::OnceLock;

use super::errors::raise;
use super::prelude::*;
use super::shape::{impl_distance_shape, GeoPath};
use super::xyz_bounds::XYZBounds;

/// The distance styles' slots in a per-style cache.
fn style_index(style: DistanceStyle) -> usize {
    match style {
        DistanceStyle::Arc => 0,
        DistanceStyle::Linear => 1,
        DistanceStyle::LinearSquared => 2,
        DistanceStyle::Normal => 3,
        DistanceStyle::NormalSquared => 4,
    }
}

/// `DistancePair`.
#[derive(Debug, Clone, Copy)]
struct DistancePair {
    path_center_distance: f64,
    distance_along_path: f64,
}

/// The components a path is made of.
#[derive(Debug, Clone)]
struct Parts {
    planet_model: Arc<PlanetModel>,
    segments: Vec<PathSegment>,
    endpoints: Vec<SegmentEndpoint>,
}

/// `PathComponent`: a node of the tree, an end point or a segment.
#[derive(Debug, Clone)]
enum Component {
    Node(Box<PathNode>),
    Endpoint(usize),
    Segment(usize),
}

/// `PathNode`: two children and their combined bounds.
#[derive(Debug, Clone)]
struct PathNode {
    child1: Component,
    child2: Component,
    bounds: XYZBounds,
}

/// `BaseSegmentEndpoint`'s subclasses.
#[derive(Debug, Clone)]
enum EndpointKind {
    /// `CircleSegmentEndpoint`: a circle around a lone point.
    Circle,
    /// `CutoffSingleCircleSegmentEndpoint`: a half circle at a path's end.
    CutoffSingle {
        cutoff_plane: SidedPlane,
        notable_points: [GeoPoint; 2],
    },
    /// `CutoffDualCircleSegmentEndpoint`: the join between two segments.
    CutoffDual {
        circle_plane2: SidedPlane,
        notable_points1: Vec<GeoPoint>,
        notable_points2: Vec<GeoPoint>,
        boundary_plane1: SidedPlane,
        boundary_plane2: SidedPlane,
    },
}

/// A `SegmentEndpoint`.
#[derive(Debug, Clone)]
struct SegmentEndpoint {
    /// The segment before this end point (`previous`).
    previous: Option<usize>,
    point: GeoPoint,
    /// `circlePlane` (`circlePlane1` for the dual kind).
    circle_plane: SidedPlane,
    kind: EndpointKind,
    /// A dual end point whose three-point circle plane could not be built:
    /// Java stores `null` and fails (`NullPointerException`) only when the
    /// path's tree first reads it, after every other component is built --
    /// so a later component's exception wins. The plane fields then hold a
    /// placeholder that is never read.
    missing_plane: bool,
}

/// A `PathSegment`.
#[derive(Debug, Clone)]
struct PathSegment {
    previous: Option<usize>,
    start: GeoPoint,
    end: GeoPoint,
    start_distance_cache: [OnceLock<f64>; 5],
    normalized_connecting_plane: Plane,
    upper_connecting_plane: SidedPlane,
    lower_connecting_plane: SidedPlane,
    start_cutoff_plane: SidedPlane,
    end_cutoff_plane: SidedPlane,
    URHC: GeoPoint,
    LRHC: GeoPoint,
    ULHC: GeoPoint,
    LLHC: GeoPoint,
    upper_connecting_plane_points: [GeoPoint; 2],
    lower_connecting_plane_points: [GeoPoint; 2],
}

/// `RuntimeException("Can't find world intersection for point ...")`.
pub(crate) fn no_world_intersection(x: f64, y: f64, z: f64) -> Error {
    use crate::geo::java_double_string as d;
    Error::Runtime(format!(
        "Can't find world intersection for point x={} y={} z={}",
        d(x),
        d(y),
        d(z)
    ))
}

/// The one point the connecting plane and its perpendicular through
/// `(x, y, z)` meet at within the segment, or the `RuntimeException` Java
/// throws when there is none (raised; `None` returned). Shared with
/// [`super::geo_degenerate_path`], whose segments do the same.
#[allow(clippy::too_many_arguments)]
pub(crate) fn perpendicular_point(
    pm: &PlanetModel,
    plane: &Plane,
    start_cutoff: &SidedPlane,
    end_cutoff: &SidedPlane,
    normalized_perp_plane: &Plane,
    x: f64,
    y: f64,
    z: f64,
) -> Option<GeoPoint> {
    let points = plane.find_intersections_two(pm, normalized_perp_plane, &[]);
    let within = |p: &GeoPoint| start_cutoff.is_within(p) && end_cutoff.is_within(p);
    let found = match points {
        [None, _] => None,
        [Some(p), None] => Some(p),
        [Some(p0), Some(p1)] => [p0, p1].into_iter().find(within),
    };
    if found.is_none() {
        raise(no_world_intersection(x, y, z));
    }
    found
}

impl PathSegment {
    fn new(
        pm: &PlanetModel,
        previous: Option<usize>,
        start: GeoPoint,
        end: GeoPoint,
        normalized_connecting_plane: Plane,
        plane_bounding_offset: f64,
    ) -> Result<PathSegment> {
        // Either start or end should be on the correct side
        let upper_connecting_plane =
            SidedPlane::from_normal(&start, &normalized_connecting_plane, -plane_bounding_offset)?;
        let lower_connecting_plane =
            SidedPlane::from_normal(&start, &normalized_connecting_plane, plane_bounding_offset)?;
        // Cutoff planes use opposite endpoints as correct side examples
        let start_cutoff_plane =
            SidedPlane::from_vectors(&end, &normalized_connecting_plane, &start)?;
        let end_cutoff_plane =
            SidedPlane::from_vectors(&start, &normalized_connecting_plane, &end)?;
        let corner = |p: &SidedPlane,
                      q: &SidedPlane,
                      side1: &SidedPlane,
                      side2: &SidedPlane|
         -> Result<GeoPoint> {
            let points = p
                .find_intersections(pm, q, &[side1, side2])
                .unwrap_or_default();
            if points.is_empty() {
                return Err(illegal(
                    "Some segment boundary points are off the ellipsoid; path too wide",
                ));
            }
            if points.len() > 1 {
                return Err(illegal("Ambiguous boundary points; path too short"));
            }
            Ok(points.into_iter().next().expect("one point"))
        };
        // Compute four points
        let ULHC = corner(
            &upper_connecting_plane,
            &start_cutoff_plane,
            &lower_connecting_plane,
            &end_cutoff_plane,
        );
        let ULHC = ULHC?;
        let URHC = corner(
            &upper_connecting_plane,
            &end_cutoff_plane,
            &lower_connecting_plane,
            &start_cutoff_plane,
        );
        let URHC = URHC?;
        let LLHC = corner(
            &lower_connecting_plane,
            &start_cutoff_plane,
            &upper_connecting_plane,
            &end_cutoff_plane,
        );
        let LLHC = LLHC?;
        let LRHC = corner(
            &lower_connecting_plane,
            &end_cutoff_plane,
            &upper_connecting_plane,
            &start_cutoff_plane,
        );
        let LRHC = LRHC?;
        Ok(PathSegment {
            previous,
            start,
            end,
            start_distance_cache: Default::default(),
            normalized_connecting_plane,
            upper_connecting_plane,
            lower_connecting_plane,
            start_cutoff_plane,
            end_cutoff_plane,
            upper_connecting_plane_points: [ULHC.clone(), URHC.clone()],
            lower_connecting_plane_points: [LLHC.clone(), LRHC.clone()],
            URHC,
            LRHC,
            ULHC,
            LLHC,
        })
    }

    fn full_path_distance(&self, style: DistanceStyle) -> f64 {
        style.to_aggregation_form(style.compute_distance(
            &self.start,
            self.end.x,
            self.end.y,
            self.end.z,
        ))
    }

    fn is_within(&self, x: f64, y: f64, z: f64) -> bool {
        self.start_cutoff_plane.is_within_xyz(x, y, z)
            && self.end_cutoff_plane.is_within_xyz(x, y, z)
            && self.upper_connecting_plane.is_within_xyz(x, y, z)
            && self.lower_connecting_plane.is_within_xyz(x, y, z)
    }

    fn is_within_section(&self, x: f64, y: f64, z: f64) -> bool {
        self.start_cutoff_plane.is_within_xyz(x, y, z)
            && self.end_cutoff_plane.is_within_xyz(x, y, z)
    }

    fn starting_distance(&self, parts: &Parts, style: DistanceStyle) -> f64 {
        *self.start_distance_cache[style_index(style)].get_or_init(|| match self.previous {
            None => 0.0,
            Some(p) => {
                let prev = &parts.segments[p];
                style.aggregate_distances(&[
                    prev.starting_distance(parts, style),
                    prev.full_path_distance(style),
                ])
            }
        })
    }

    fn distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        let starting_distance = self.starting_distance(parts, style);
        let path_distance = self.path_distance(parts, style, x, y, z);
        style.from_aggregation_form(style.aggregate_distances(&[starting_distance, path_distance]))
    }

    fn nearest_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<DistancePair> {
        if !self.is_within_section(x, y, z) {
            return None;
        }
        Some(DistancePair {
            path_center_distance: self.path_center_distance(parts, style, x, y, z),
            distance_along_path: style.aggregate_distances(&[
                self.starting_distance(parts, style),
                self.nearest_path_distance(parts, style, x, y, z),
            ]),
        })
    }

    /// The normalized plane through the origin and `(x, y, z)`
    /// perpendicular to the connecting plane, or `None` when the point is
    /// on the plane's normal.
    fn perp_plane(&self, x: f64, y: f64, z: f64) -> Option<Plane> {
        let n = &self.normalized_connecting_plane;
        let perp_x = n.y * z - n.z * y;
        let perp_y = n.z * x - n.x * z;
        let perp_z = n.x * y - n.y * x;
        let magnitude = sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z);
        if abs(magnitude) < MINIMUM_RESOLUTION {
            return None;
        }
        let norm_factor = 1.0 / magnitude;
        Some(Plane::new(
            perp_x * norm_factor,
            perp_y * norm_factor,
            perp_z * norm_factor,
            0.0,
        ))
    }

    fn the_point(
        &self,
        pm: &PlanetModel,
        perp: &Plane,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<GeoPoint> {
        perpendicular_point(
            pm,
            &self.normalized_connecting_plane,
            &self.start_cutoff_plane,
            &self.end_cutoff_plane,
            perp,
            x,
            y,
            z,
        )
    }

    fn path_center_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        // First, if this point is outside the endplanes of the segment,
        // return POSITIVE_INFINITY.
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        // (1) Compute normalizedPerpPlane.  If degenerate, then there is no
        // such plane, which means that the point given is insufficient to
        // distinguish between a family of such planes. This can happen only if
        // the point is one of the "poles", imagining the normalized plane to
        // be the "equator".  In that case, the distance returned should be
        // zero. Want no allocations or expensive operations!  So we do this
        // the hard way
        let Some(perp) = self.perp_plane(x, y, z) else {
            return style.compute_distance(&self.start, x, y, z);
        };
        let Some(the_point) = self.the_point(&parts.planet_model, &perp, x, y, z) else {
            return f64::INFINITY;
        };
        style.to_aggregation_form(style.compute_distance(&the_point, x, y, z))
    }

    fn nearest_path_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        let Some(perp) = self.perp_plane(x, y, z) else {
            return style.to_aggregation_form(0.0);
        };
        let Some(the_point) = self.the_point(&parts.planet_model, &perp, x, y, z) else {
            return f64::INFINITY;
        };
        style.to_aggregation_form(style.compute_distance(
            &self.start,
            the_point.x,
            the_point.y,
            the_point.z,
        ))
    }

    fn path_delta_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        let Some(perp) = self.perp_plane(x, y, z) else {
            let the_distance = style.compute_distance(&self.start, x, y, z);
            return style.aggregate_distances(&[the_distance, the_distance]);
        };
        let Some(the_point) = self.the_point(&parts.planet_model, &perp, x, y, z) else {
            return f64::INFINITY;
        };
        let the_distance = style.to_aggregation_form(style.compute_distance(&the_point, x, y, z));
        style.aggregate_distances(&[the_distance, the_distance])
    }

    fn path_distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        let Some(perp) = self.perp_plane(x, y, z) else {
            return style.to_aggregation_form(style.compute_distance(&self.start, x, y, z));
        };
        let Some(the_point) = self.the_point(&parts.planet_model, &perp, x, y, z) else {
            return f64::INFINITY;
        };
        style.aggregate_distances(&[
            style.to_aggregation_form(style.compute_distance(&the_point, x, y, z)),
            style.to_aggregation_form(style.compute_distance(
                &self.start,
                the_point.x,
                the_point.y,
                the_point.z,
            )),
        ])
    }

    fn outside_distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*parts.planet_model;
        let upper_distance = style.compute_distance_to_plane(
            pm,
            &self.upper_connecting_plane,
            x,
            y,
            z,
            &[
                &self.lower_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        );
        let lower_distance = style.compute_distance_to_plane(
            pm,
            &self.lower_connecting_plane,
            x,
            y,
            z,
            &[
                &self.upper_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        );
        let start_distance = style.compute_distance_to_plane(
            pm,
            &self.start_cutoff_plane,
            x,
            y,
            z,
            &[
                &self.end_cutoff_plane,
                &self.lower_connecting_plane,
                &self.upper_connecting_plane,
            ],
        );
        let end_distance = style.compute_distance_to_plane(
            pm,
            &self.end_cutoff_plane,
            x,
            y,
            z,
            &[
                &self.start_cutoff_plane,
                &self.lower_connecting_plane,
                &self.upper_connecting_plane,
            ],
        );
        let ulhc_distance = style.compute_distance(&self.ULHC, x, y, z);
        let urhc_distance = style.compute_distance(&self.URHC, x, y, z);
        let llhc_distance = style.compute_distance(&self.LLHC, x, y, z);
        let lrhc_distance = style.compute_distance(&self.LRHC, x, y, z);
        style.to_aggregation_form(min(
            min(
                min(upper_distance, lower_distance),
                min(start_distance, end_distance),
            ),
            min(
                min(ulhc_distance, urhc_distance),
                min(llhc_distance, lrhc_distance),
            ),
        ))
    }

    fn intersects(
        &self,
        pm: &PlanetModel,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        self.upper_connecting_plane.intersects(
            pm,
            p,
            notable_points,
            &self.upper_connecting_plane_points,
            bounds,
            &[
                &self.lower_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        ) || self.lower_connecting_plane.intersects(
            pm,
            p,
            notable_points,
            &self.lower_connecting_plane_points,
            bounds,
            &[
                &self.upper_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        )
    }

    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.upper_connecting_plane,
            &self.upper_connecting_plane_points,
            &[
                &self.lower_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        ) || geo_shape.intersects(
            &self.lower_connecting_plane,
            &self.lower_connecting_plane_points,
            &[
                &self.upper_connecting_plane,
                &self.start_cutoff_plane,
                &self.end_cutoff_plane,
            ],
        )
    }

    fn get_bounds(&self, pm: &PlanetModel, bounds: &mut dyn Bounds) {
        // We need to do all bounding planes as well as corner points
        base_get_bounds(&SegmentMembership(self), pm, bounds);
        let (u, l, s, e) = (
            &self.upper_connecting_plane,
            &self.lower_connecting_plane,
            &self.start_cutoff_plane,
            &self.end_cutoff_plane,
        );
        bounds
            .add_point(&self.start)
            .add_point(&self.end)
            .add_point(&self.ULHC)
            .add_point(&self.URHC)
            .add_point(&self.LRHC)
            .add_point(&self.LLHC)
            .add_plane(pm, u, &[l, s, e])
            .add_plane(pm, l, &[u, s, e])
            .add_plane(pm, s, &[e, u, l])
            .add_plane(pm, e, &[s, u, l])
            .add_intersection(pm, u, s, &[l, e])
            .add_intersection(pm, s, l, &[e, u])
            .add_intersection(pm, l, e, &[u, s])
            .add_intersection(pm, e, u, &[s, l]);
    }
}

/// A segment as the `Membership` `GeoBaseBounds.getBounds` tests the poles
/// against.
struct SegmentMembership<'a>(&'a PathSegment);

impl Membership for SegmentMembership<'_> {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.0.is_within(x, y, z)
    }
}

/// An end point as the `Membership` `GeoBaseBounds.getBounds` tests.
struct EndpointMembership<'a>(&'a SegmentEndpoint);

impl Membership for EndpointMembership<'_> {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.0.is_within(x, y, z)
    }
}

impl SegmentEndpoint {
    /// `CircleSegmentEndpoint(planetModel, previous, point, normalPlane,
    /// upperPoint, lowerPoint)`.
    fn circle(
        previous: Option<usize>,
        point: GeoPoint,
        normal_plane: &Plane,
        upper_point: &GeoPoint,
        lower_point: &GeoPoint,
    ) -> Result<SegmentEndpoint> {
        let circle_plane = SidedPlane::construct_normalized_perpendicular_sided_plane(
            &point,
            normal_plane,
            upper_point,
            lower_point,
        )?
        .ok_or_else(|| Error::NullPointer("could not construct the circle plane".into()))?;
        Ok(SegmentEndpoint {
            previous,
            point,
            circle_plane,
            kind: EndpointKind::Circle,
            missing_plane: false,
        })
    }

    /// `CutoffSingleCircleSegmentEndpoint(planetModel, previous, point,
    /// cutoffPlane, topEdgePoint, bottomEdgePoint)`.
    fn cutoff_single(
        previous: Option<usize>,
        point: GeoPoint,
        cutoff_plane: &SidedPlane,
        top_edge_point: &GeoPoint,
        bottom_edge_point: &GeoPoint,
    ) -> Result<SegmentEndpoint> {
        let circle_plane = SidedPlane::construct_sided_plane_from_two_points(
            &point,
            top_edge_point,
            bottom_edge_point,
        );
        let circle_plane = circle_plane?;
        Ok(SegmentEndpoint {
            previous,
            point,
            circle_plane,
            kind: EndpointKind::CutoffSingle {
                cutoff_plane: SidedPlane::opposite(cutoff_plane),
                notable_points: [top_edge_point.clone(), bottom_edge_point.clone()],
            },
            missing_plane: false,
        })
    }

    /// `CutoffDualCircleSegmentEndpoint(...)`.
    #[allow(clippy::too_many_arguments)]
    fn cutoff_dual(
        previous: Option<usize>,
        point: GeoPoint,
        prev_cutoff_plane: &SidedPlane,
        next_cutoff_plane: &SidedPlane,
        prev_URHC: &GeoPoint,
        prev_LRHC: &GeoPoint,
        current_ULHC: &GeoPoint,
        current_LLHC: &GeoPoint,
    ) -> Result<SegmentEndpoint> {
        let missing = std::cell::Cell::new(false);
        let three = |a: &GeoPoint, b: &GeoPoint, c: &GeoPoint| -> Result<SidedPlane> {
            let plane = SidedPlane::construct_normalized_three_point_sided_plane(&point, a, b, c);
            missing.set(missing.get() || plane.is_none());
            Ok(plane.unwrap_or(*prev_cutoff_plane))
        };
        // Note: What we really need is a single plane that goes through all
        // four points. Since that's not possible in the ellipsoid
        // implementation (because three points determine a plane, not four),
        // we need two planes. BUT we need to be careful to not cut off any
        // points from the circle that would be within the path otherwise.
        let (circle_plane1, notable_points1) = if !prev_cutoff_plane.is_within(current_ULHC) {
            (
                three(prev_URHC, prev_LRHC, current_ULHC)?,
                vec![prev_URHC.clone(), prev_LRHC.clone(), current_ULHC.clone()],
            )
        } else if !prev_cutoff_plane.is_within(current_LLHC) {
            (
                three(prev_URHC, prev_LRHC, current_LLHC)?,
                vec![prev_URHC.clone(), prev_LRHC.clone(), current_LLHC.clone()],
            )
        } else {
            (
                SidedPlane::construct_sided_plane_from_two_points(&point, prev_URHC, prev_LRHC)?,
                vec![prev_URHC.clone(), prev_LRHC.clone()],
            )
        };
        let (circle_plane2, notable_points2) = if !next_cutoff_plane.is_within(prev_URHC) {
            (
                three(current_ULHC, current_LLHC, prev_URHC)?,
                vec![
                    current_ULHC.clone(),
                    current_LLHC.clone(),
                    prev_URHC.clone(),
                ],
            )
        } else if !next_cutoff_plane.is_within(prev_LRHC) {
            (
                three(current_ULHC, current_LLHC, prev_LRHC)?,
                vec![
                    current_ULHC.clone(),
                    current_LLHC.clone(),
                    prev_LRHC.clone(),
                ],
            )
        } else {
            (
                SidedPlane::construct_sided_plane_from_two_points(
                    &point,
                    current_ULHC,
                    current_LLHC,
                )?,
                vec![current_ULHC.clone(), current_LLHC.clone()],
            )
        };
        Ok(SegmentEndpoint {
            previous,
            point,
            circle_plane: circle_plane1,
            kind: EndpointKind::CutoffDual {
                circle_plane2,
                notable_points1,
                notable_points2,
                boundary_plane1: SidedPlane::opposite(prev_cutoff_plane),
                boundary_plane2: SidedPlane::opposite(next_cutoff_plane),
            },
            missing_plane: missing.get(),
        })
    }

    fn is_within(&self, x: f64, y: f64, z: f64) -> bool {
        match &self.kind {
            EndpointKind::Circle => self.circle_plane.is_within_xyz(x, y, z),
            EndpointKind::CutoffSingle { cutoff_plane, .. } => {
                cutoff_plane.is_within_xyz(x, y, z) && self.circle_plane.is_within_xyz(x, y, z)
            }
            EndpointKind::CutoffDual {
                circle_plane2,
                boundary_plane1,
                boundary_plane2,
                ..
            } => {
                if !boundary_plane1.is_within_xyz(x, y, z)
                    || !boundary_plane2.is_within_xyz(x, y, z)
                {
                    return false;
                }
                self.circle_plane.is_within_xyz(x, y, z) || circle_plane2.is_within_xyz(x, y, z)
            }
        }
    }

    fn is_within_section(&self, x: f64, y: f64, z: f64) -> bool {
        match &self.kind {
            EndpointKind::Circle => true,
            EndpointKind::CutoffSingle { cutoff_plane, .. } => cutoff_plane.is_within_xyz(x, y, z),
            EndpointKind::CutoffDual {
                boundary_plane1,
                boundary_plane2,
                ..
            } => boundary_plane1.is_within_xyz(x, y, z) && boundary_plane2.is_within_xyz(x, y, z),
        }
    }

    fn starting_distance(&self, parts: &Parts, style: DistanceStyle) -> f64 {
        match self.previous {
            None => style.to_aggregation_form(0.0),
            Some(p) => {
                let prev = &parts.segments[p];
                style.aggregate_distances(&[
                    prev.starting_distance(parts, style),
                    prev.full_path_distance(style),
                ])
            }
        }
    }

    fn distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        let starting_distance = self.starting_distance(parts, style);
        let path_distance = self.path_distance(style, x, y, z);
        style.aggregate_distances(&[starting_distance, path_distance])
    }

    fn nearest_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<DistancePair> {
        if !self.is_within_section(x, y, z) {
            return None;
        }
        Some(DistancePair {
            path_center_distance: self.path_center_distance(style, x, y, z),
            distance_along_path: style.aggregate_distances(&[
                self.starting_distance(parts, style),
                self.nearest_path_distance(style, x, y, z),
            ]),
        })
    }

    // `fullPathDistance` (0 for an end point) is only read by
    // `PathNode.pathDistance`, which nothing in `GeoStandardPath` calls; it
    // is not ported.

    fn path_delta_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        let the_distance = style.to_aggregation_form(style.compute_distance(&self.point, x, y, z));
        style.aggregate_distances(&[the_distance, the_distance])
    }

    fn path_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        style.to_aggregation_form(style.compute_distance(&self.point, x, y, z))
    }

    fn nearest_path_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        style.to_aggregation_form(0.0)
    }

    fn path_center_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        style.to_aggregation_form(style.compute_distance(&self.point, x, y, z))
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.to_aggregation_form(style.compute_distance(&self.point, x, y, z))
    }

    fn intersects(
        &self,
        pm: &PlanetModel,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        match &self.kind {
            EndpointKind::Circle => {
                self.circle_plane
                    .intersects(pm, p, notable_points, &[], bounds, &[])
            }
            EndpointKind::CutoffSingle {
                cutoff_plane,
                notable_points: own,
            } => self
                .circle_plane
                .intersects(pm, p, notable_points, own, bounds, &[cutoff_plane]),
            EndpointKind::CutoffDual {
                circle_plane2,
                notable_points1,
                notable_points2,
                boundary_plane1,
                boundary_plane2,
            } => {
                self.circle_plane.intersects(
                    pm,
                    p,
                    notable_points,
                    notable_points1,
                    bounds,
                    &[boundary_plane1, boundary_plane2],
                ) || circle_plane2.intersects(
                    pm,
                    p,
                    notable_points,
                    notable_points2,
                    bounds,
                    &[boundary_plane1, boundary_plane2],
                )
            }
        }
    }

    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        match &self.kind {
            EndpointKind::Circle => geo_shape.intersects(&self.circle_plane, &[], &[]),
            EndpointKind::CutoffSingle {
                cutoff_plane,
                notable_points,
            } => geo_shape.intersects(&self.circle_plane, notable_points, &[cutoff_plane]),
            EndpointKind::CutoffDual {
                circle_plane2,
                notable_points1,
                notable_points2,
                boundary_plane1,
                boundary_plane2,
            } => {
                geo_shape.intersects(
                    &self.circle_plane,
                    notable_points1,
                    &[boundary_plane1, boundary_plane2],
                ) || geo_shape.intersects(
                    circle_plane2,
                    notable_points2,
                    &[boundary_plane1, boundary_plane2],
                )
            }
        }
    }

    fn get_bounds(&self, pm: &PlanetModel, bounds: &mut dyn Bounds) {
        // BaseSegmentEndpoint
        base_get_bounds(&EndpointMembership(self), pm, bounds);
        bounds.add_point(&self.point);
        let c1 = &self.circle_plane;
        match &self.kind {
            EndpointKind::Circle => {
                bounds.add_plane(pm, c1, &[]);
            }
            EndpointKind::CutoffSingle { cutoff_plane, .. } => {
                // CircleSegmentEndpoint, then its own
                bounds.add_plane(pm, c1, &[]);
                bounds.add_plane(pm, c1, &[cutoff_plane]);
                bounds.add_plane(pm, cutoff_plane, &[c1]);
                bounds.add_intersection(pm, c1, cutoff_plane, &[]);
                bounds.add_intersection(pm, cutoff_plane, c1, &[]);
            }
            EndpointKind::CutoffDual {
                circle_plane2: c2,
                boundary_plane1: b1,
                boundary_plane2: b2,
                ..
            } => {
                bounds.add_plane(pm, c1, &[b1, b2]);
                bounds.add_plane(pm, c2, &[b1, b2]);
                bounds.add_plane(pm, b1, &[c1, b2]);
                bounds.add_plane(pm, b1, &[c2, b2]);
                bounds.add_plane(pm, b2, &[c1, b1]);
                bounds.add_plane(pm, b2, &[c2, b1]);
                bounds.add_intersection(pm, c1, b1, &[b2]);
                bounds.add_intersection(pm, c1, b2, &[b1]);
                bounds.add_intersection(pm, c2, b1, &[b2]);
                bounds.add_intersection(pm, c2, b2, &[b1]);
            }
        }
    }
}

impl Component {
    fn get_bounds(&self, parts: &Parts, bounds: &mut dyn Bounds) {
        match self {
            Component::Node(n) => match bounds.as_xyz_bounds() {
                Some(xyz) => n.bounds.add_bounds(xyz),
                None => {
                    n.child1.get_bounds(parts, bounds);
                    n.child2.get_bounds(parts, bounds);
                }
            },
            Component::Endpoint(i) => parts.endpoints[*i].get_bounds(&parts.planet_model, bounds),
            Component::Segment(i) => parts.segments[*i].get_bounds(&parts.planet_model, bounds),
        }
    }

    fn is_within(&self, parts: &Parts, x: f64, y: f64, z: f64) -> bool {
        match self {
            Component::Node(n) => {
                let b = &n.bounds;
                // Java compares against the unboxed bounds; a node's bounds
                // always hold every child's points.
                let out = |v: f64, lo: Option<f64>, hi: Option<f64>| {
                    v < lo.unwrap_or(f64::NAN) || v > hi.unwrap_or(f64::NAN)
                };
                if out(x, b.minimum_x(), b.maximum_x())
                    || out(y, b.minimum_y(), b.maximum_y())
                    || out(z, b.minimum_z(), b.maximum_z())
                {
                    return false;
                }
                n.child1.is_within(parts, x, y, z) || n.child2.is_within(parts, x, y, z)
            }
            Component::Endpoint(i) => parts.endpoints[*i].is_within(x, y, z),
            Component::Segment(i) => parts.segments[*i].is_within(x, y, z),
        }
    }

    fn distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        match self {
            Component::Node(n) => {
                if !self.is_within(parts, x, y, z) {
                    return f64::INFINITY;
                }
                let child1_distance = n.child1.distance(parts, style, x, y, z);
                let child2_distance = n.child2.distance(parts, style, x, y, z);
                min(child1_distance, child2_distance)
            }
            Component::Endpoint(i) => parts.endpoints[*i].distance(parts, style, x, y, z),
            Component::Segment(i) => parts.segments[*i].distance(parts, style, x, y, z),
        }
    }

    fn nearest_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> Option<DistancePair> {
        match self {
            Component::Node(n) => {
                let first = n.child1.nearest_distance(parts, style, x, y, z);
                let second = n.child2.nearest_distance(parts, style, x, y, z);
                let (Some(f), Some(s)) = (first, second) else {
                    return first.or(second);
                };
                if f.path_center_distance < s.path_center_distance {
                    Some(f)
                } else if s.path_center_distance < f.path_center_distance {
                    Some(s)
                } else if f.distance_along_path < s.distance_along_path {
                    Some(f)
                } else {
                    Some(s)
                }
            }
            Component::Endpoint(i) => parts.endpoints[*i].nearest_distance(parts, style, x, y, z),
            Component::Segment(i) => parts.segments[*i].nearest_distance(parts, style, x, y, z),
        }
    }

    fn path_delta_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        match self {
            Component::Node(n) => {
                if !self.is_within(parts, x, y, z) {
                    return f64::INFINITY;
                }
                min(
                    n.child1.path_delta_distance(parts, style, x, y, z),
                    n.child2.path_delta_distance(parts, style, x, y, z),
                )
            }
            Component::Endpoint(i) => parts.endpoints[*i].path_delta_distance(style, x, y, z),
            Component::Segment(i) => parts.segments[*i].path_delta_distance(parts, style, x, y, z),
        }
    }

    fn path_center_distance(
        &self,
        parts: &Parts,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        match self {
            Component::Node(n) => min(
                n.child1.path_center_distance(parts, style, x, y, z),
                n.child2.path_center_distance(parts, style, x, y, z),
            ),
            Component::Endpoint(i) => parts.endpoints[*i].path_center_distance(style, x, y, z),
            Component::Segment(i) => parts.segments[*i].path_center_distance(parts, style, x, y, z),
        }
    }

    fn outside_distance(&self, parts: &Parts, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        match self {
            Component::Node(n) => min(
                n.child1.outside_distance(parts, style, x, y, z),
                n.child2.outside_distance(parts, style, x, y, z),
            ),
            Component::Endpoint(i) => parts.endpoints[*i].outside_distance(style, x, y, z),
            Component::Segment(i) => parts.segments[*i].outside_distance(parts, style, x, y, z),
        }
    }

    fn intersects(
        &self,
        parts: &Parts,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        // `planeBounds` is always null (GeoStandardPath.intersects).
        match self {
            Component::Node(n) => {
                n.child1.intersects(parts, p, notable_points, bounds)
                    || n.child2.intersects(parts, p, notable_points, bounds)
            }
            Component::Endpoint(i) => {
                parts.endpoints[*i].intersects(&parts.planet_model, p, notable_points, bounds)
            }
            Component::Segment(i) => {
                parts.segments[*i].intersects(&parts.planet_model, p, notable_points, bounds)
            }
        }
    }

    fn intersects_shape(&self, parts: &Parts, geo_shape: &dyn GeoShape) -> bool {
        match self {
            Component::Node(n) => {
                n.child1.intersects_shape(parts, geo_shape)
                    || n.child2.intersects_shape(parts, geo_shape)
            }
            Component::Endpoint(i) => parts.endpoints[*i].intersects_shape(geo_shape),
            Component::Segment(i) => parts.segments[*i].intersects_shape(geo_shape),
        }
    }
}

/// `TreeBuilder`: pairs components bottom-up into a balanced tree.
struct TreeBuilder {
    component_stack: Vec<Component>,
    depth_stack: Vec<i32>,
}

impl TreeBuilder {
    fn add_component(&mut self, parts: &Parts, component: Component) {
        self.component_stack.push(component);
        self.depth_stack.push(0);
        while self.depth_stack.len() >= 2 {
            let n = self.depth_stack.len();
            if self.depth_stack[n - 1] == self.depth_stack[n - 2] {
                self.merge_top(parts);
            } else {
                break;
            }
        }
    }

    fn root(mut self, parts: &Parts) -> Option<Component> {
        if self.component_stack.is_empty() {
            return None;
        }
        while self.component_stack.len() > 1 {
            self.merge_top(parts);
        }
        self.component_stack.pop()
    }

    fn merge_top(&mut self, parts: &Parts) {
        self.depth_stack.pop();
        let second = self.component_stack.pop().expect("two components");
        let new_depth = self.depth_stack.pop().expect("two depths") + 1;
        let first = self.component_stack.pop().expect("two components");
        self.depth_stack.push(new_depth);
        let mut bounds = XYZBounds::new();
        first.get_bounds(parts, &mut bounds);
        second.get_bounds(parts, &mut bounds);
        self.component_stack
            .push(Component::Node(Box::new(PathNode {
                child1: first,
                child2: second,
                bounds,
            })));
    }
}

/// A path.
#[derive(Debug, Clone)]
pub struct GeoStandardPath {
    planet_model: Arc<PlanetModel>,
    cutoff_angle: f64,
    points: Vec<GeoPoint>,
    parts: Parts,
    root_component: Option<Component>,
    edge_points: Vec<GeoPoint>,
}

impl GeoStandardPath {
    /// `GeoStandardPath(planetModel, maxCutoffAngle, pathPoints)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        max_cutoff_angle: f64,
        path_points: &[GeoPoint],
    ) -> Result<GeoStandardPath> {
        let pm = &**planet_model;
        if max_cutoff_angle <= 0.0 || max_cutoff_angle > PI * 0.5 {
            return Err(illegal("Cutoff angle out of bounds"));
        }
        let cutoff_angle = max_cutoff_angle;
        let sin_angle = sin(max_cutoff_angle);
        // done()
        let points: Vec<GeoPoint> = path_points.to_vec();
        if points.is_empty() {
            return Err(illegal("Path must have at least one point"));
        }
        let mut parts = Parts {
            planet_model: planet_model.clone(),
            segments: Vec::with_capacity(points.len()),
            endpoints: Vec::with_capacity(points.len()),
        };
        // Compute an offset to use for all segments.  This will be based on
        // the minimum magnitude of the entire ellipsoid.
        let cutoff_offset = sin_angle * pm.minimum_magnitude();
        // First, build all segments.  We'll then go back and build
        // corresponding segment endpoints.
        let mut last_point: Option<&GeoPoint> = None;
        let mut last_component: Option<usize> = None;
        for end in &points {
            if let Some(lp) = last_point {
                let normalized_connecting_plane = Plane::from_vectors(lp, end)?;
                let new_component = PathSegment::new(
                    pm,
                    last_component,
                    lp.clone(),
                    end.clone(),
                    normalized_connecting_plane,
                    cutoff_offset,
                );
                let new_component = new_component?;
                parts.segments.push(new_component);
                last_component = Some(parts.segments.len() - 1);
            }
            last_point = Some(end);
        }
        let edge_points;
        if parts.segments.is_empty() {
            // Simple circle
            let point = &points[0];
            let lat = point.latitude();
            let lon = point.longitude();
            // Compute two points on the circle, with the right angle from the
            // center.  We'll use these to obtain the perpendicular plane to
            // the circle.
            let mut upper_lat = lat + cutoff_angle;
            let mut upper_lon = lon;
            if upper_lat > PI * 0.5 {
                upper_lon += PI;
                if upper_lon > PI {
                    upper_lon -= 2.0 * PI;
                }
                upper_lat = PI - upper_lat;
            }
            let mut lower_lat = lat - cutoff_angle;
            let mut lower_lon = lon;
            if lower_lat < -PI * 0.5 {
                lower_lon += PI;
                if lower_lon > PI {
                    lower_lon -= 2.0 * PI;
                }
                lower_lat = -PI - lower_lat;
            }
            let upper_point = GeoPoint::from_lat_lon(pm, upper_lat, upper_lon)?;
            let lower_point = GeoPoint::from_lat_lon(pm, lower_lat, lower_lon)?;
            // Construct normal plane
            let normal_plane =
                Plane::construct_normalized_z_plane_points(&[&upper_point, &lower_point, point])
                    .ok_or_else(|| Error::NullPointer("all points are on the z axis".into()))?;
            let only_endpoint = SegmentEndpoint::circle(
                None,
                point.clone(),
                &normal_plane,
                &upper_point,
                &lower_point,
            );
            let only_endpoint = only_endpoint?;
            let sample = only_endpoint
                .circle_plane
                .sample_intersection_point(pm, &normal_plane)
                .ok_or_else(|| illegal("no edge point (null edge point in Java)"))?;
            parts.endpoints.push(only_endpoint);
            edge_points = vec![sample];
        } else {
            // Create segment endpoints.  Use an appropriate constructor for
            // the start and end of the path.
            let mut first_edge = None;
            for i in 0..parts.segments.len() {
                let current = &parts.segments[i];
                if i == 0 {
                    // Starting endpoint
                    let start_endpoint = SegmentEndpoint::cutoff_single(
                        None,
                        current.start.clone(),
                        &current.start_cutoff_plane,
                        &current.ULHC,
                        &current.LLHC,
                    );
                    let start_endpoint = start_endpoint?;
                    first_edge = Some(current.ULHC.clone());
                    parts.endpoints.push(start_endpoint);
                    continue;
                }
                // General intersection case
                let prev = &parts.segments[i - 1];
                let e = SegmentEndpoint::cutoff_dual(
                    Some(i - 1),
                    current.start.clone(),
                    &prev.end_cutoff_plane,
                    &current.start_cutoff_plane,
                    &prev.URHC,
                    &prev.LRHC,
                    &current.ULHC,
                    &current.LLHC,
                );
                let e = e?;
                parts.endpoints.push(e);
            }
            // Do final endpoint
            let last = parts.segments.len() - 1;
            let last_segment = &parts.segments[last];
            let e = SegmentEndpoint::cutoff_single(
                Some(last),
                last_segment.end.clone(),
                &last_segment.end_cutoff_plane,
                &last_segment.URHC,
                &last_segment.LRHC,
            );
            let e = e?;
            parts.endpoints.push(e);
            edge_points = first_edge.into_iter().collect();
        }
        // Java builds the tree next, and its bounds dereference a null
        // circle plane.
        if parts.endpoints.iter().any(|e| e.missing_plane) {
            return Err(Error::NullPointer(
                "could not construct a circle plane".into(),
            ));
        }
        let mut tree_builder = TreeBuilder {
            component_stack: Vec::with_capacity(parts.segments.len() + parts.endpoints.len()),
            depth_stack: Vec::with_capacity(parts.segments.len() + parts.endpoints.len()),
        };
        tree_builder.add_component(&parts, Component::Endpoint(0));
        for i in 0..parts.segments.len() {
            tree_builder.add_component(&parts, Component::Segment(i));
            tree_builder.add_component(&parts, Component::Endpoint(i + 1));
        }
        let root_component = tree_builder.root(&parts);
        Ok(GeoStandardPath {
            planet_model: planet_model.clone(),
            cutoff_angle,
            points,
            parts,
            root_component,
            edge_points,
        })
    }

    /// `GeoStandardPath(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<GeoStandardPath> {
        let cutoff = read_double(input)?;
        let points = super::standard_objects::read_point_array(input)?;
        GeoStandardPath::new(planet_model, cutoff, &points)
    }

    fn distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        // Algorithm: (1) If the point is within any of the segments along the
        // path, return that value. (2) If the point is within any of the
        // segment end circles along the path, return that value.
        match &self.root_component {
            None => f64::INFINITY,
            Some(r) => style.from_aggregation_form(r.distance(&self.parts, style, x, y, z)),
        }
    }

    fn delta_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        match &self.root_component {
            None => f64::INFINITY,
            Some(r) => {
                style.from_aggregation_form(r.path_delta_distance(&self.parts, style, x, y, z))
            }
        }
    }

    fn distance_bounds(
        &self,
        bounds: &mut dyn Bounds,
        _style: DistanceStyle,
        _distance_value: f64,
    ) -> Result<()> {
        // TBD: Compute actual bounds based on distance
        self.get_bounds(bounds);
        Ok(())
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        match &self.root_component {
            None => f64::INFINITY,
            Some(r) => style.from_aggregation_form(r.outside_distance(&self.parts, style, x, y, z)),
        }
    }
}

impl SerializableObject for GeoStandardPath {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.cutoff_angle);
        super::standard_objects::write_point_array(out, &self.points);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(3)
    }
}

impl_planet_object!(GeoStandardPath);
impl_membership_shape!(GeoStandardPath);
impl_base_area!(GeoStandardPath);
impl_distance_shape!(GeoStandardPath);

impl Membership for GeoStandardPath {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        match &self.root_component {
            None => false,
            Some(r) => r.is_within(&self.parts, x, y, z),
        }
    }
}

impl Bounded for GeoStandardPath {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        base_get_bounds(self, &self.planet_model, bounds);
        if let Some(r) = &self.root_component {
            r.get_bounds(&self.parts, bounds);
        }
    }
}

impl GeoShape for GeoStandardPath {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        plane: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        match &self.root_component {
            None => false,
            Some(r) => r.intersects(&self.parts, plane, notable_points, bounds),
        }
    }
}

impl GeoAreaShape for GeoStandardPath {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        match &self.root_component {
            None => false,
            Some(r) => r.intersects_shape(&self.parts, geo_shape),
        }
    }
}

impl GeoPath for GeoStandardPath {
    fn compute_nearest_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let Some(r) = &self.root_component else {
            return f64::INFINITY;
        };
        match r.nearest_distance(&self.parts, style, x, y, z) {
            None => f64::INFINITY,
            Some(pair) => style.from_aggregation_form(pair.distance_along_path),
        }
    }

    fn compute_path_center_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        match &self.root_component {
            None => f64::INFINITY,
            Some(r) => {
                style.from_aggregation_form(r.path_center_distance(&self.parts, style, x, y, z))
            }
        }
    }
}
