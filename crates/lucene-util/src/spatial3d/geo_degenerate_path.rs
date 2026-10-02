//! `GeoDegeneratePath` (`org.apache.lucene.spatial3d.geom.GeoDegeneratePath`):
//! a path of zero width -- the great-circle segments themselves.

use std::sync::OnceLock;

use super::prelude::*;
use super::shape::{impl_distance_shape, GeoPath};

/// A `SegmentEndpoint` of a degenerate path: the point, and the cutoff
/// planes of the segments it joins.
#[derive(Debug, Clone)]
struct SegmentEndpoint {
    point: GeoPoint,
    cutoff_planes: Vec<SidedPlane>,
}

impl SegmentEndpoint {
    fn is_within(&self, x: f64, y: f64, z: f64) -> bool {
        self.point.is_identical_xyz(x, y, z)
    }

    fn is_within_section(&self, x: f64, y: f64, z: f64) -> bool {
        self.cutoff_planes.iter().all(|m| m.is_within_xyz(x, y, z))
    }

    fn path_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        style.to_aggregation_form(style.compute_distance(&self.point, x, y, z))
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

    fn intersects(&self, p: &Plane, bounds: &[&dyn Membership]) -> bool {
        // If not on the plane, no intersection
        if !p.evaluate_is_zero(&self.point) {
            return false;
        }
        bounds.iter().all(|m| m.is_within(&self.point))
    }
}

/// A `PathSegment` of a degenerate path.
#[derive(Debug, Clone)]
struct PathSegment {
    start: GeoPoint,
    end: GeoPoint,
    full_distance_cache: [OnceLock<f64>; 5],
    normalized_connecting_plane: Plane,
    start_cutoff_plane: SidedPlane,
    end_cutoff_plane: SidedPlane,
    connecting_plane_points: [GeoPoint; 2],
}

impl PathSegment {
    fn new(
        start: GeoPoint,
        end: GeoPoint,
        normalized_connecting_plane: Plane,
    ) -> Result<PathSegment> {
        // Cutoff planes use opposite endpoints as correct side examples
        let start_cutoff_plane =
            SidedPlane::from_vectors(&end, &normalized_connecting_plane, &start)?;
        let end_cutoff_plane =
            SidedPlane::from_vectors(&start, &normalized_connecting_plane, &end)?;
        Ok(PathSegment {
            connecting_plane_points: [start.clone(), end.clone()],
            start,
            end,
            full_distance_cache: Default::default(),
            normalized_connecting_plane,
            start_cutoff_plane,
            end_cutoff_plane,
        })
    }

    fn full_path_distance(&self, style: DistanceStyle) -> f64 {
        let i = match style {
            DistanceStyle::Arc => 0,
            DistanceStyle::Linear => 1,
            DistanceStyle::LinearSquared => 2,
            DistanceStyle::Normal => 3,
            DistanceStyle::NormalSquared => 4,
        };
        *self.full_distance_cache[i].get_or_init(|| {
            style.to_aggregation_form(style.compute_distance(
                &self.start,
                self.end.x,
                self.end.y,
                self.end.z,
            ))
        })
    }

    fn is_within(&self, x: f64, y: f64, z: f64) -> bool {
        self.start_cutoff_plane.is_within_xyz(x, y, z)
            && self.end_cutoff_plane.is_within_xyz(x, y, z)
            && self
                .normalized_connecting_plane
                .evaluate_is_zero_xyz(x, y, z)
    }

    fn is_within_section(&self, x: f64, y: f64, z: f64) -> bool {
        self.start_cutoff_plane.is_within_xyz(x, y, z)
            && self.end_cutoff_plane.is_within_xyz(x, y, z)
    }

    /// The point on the segment nearest `(x, y, z)`; `Err(false)` when the
    /// point is on the plane's normal (the caller's degenerate case), and
    /// `Err(true)` when there is none (Java's `RuntimeException`, raised).
    fn the_point(
        &self,
        pm: &PlanetModel,
        x: f64,
        y: f64,
        z: f64,
    ) -> std::result::Result<GeoPoint, bool> {
        let n = &self.normalized_connecting_plane;
        let perp_x = n.y * z - n.z * y;
        let perp_y = n.z * x - n.x * z;
        let perp_z = n.x * y - n.y * x;
        let magnitude = sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z);
        if abs(magnitude) < MINIMUM_RESOLUTION {
            return Err(false);
        }
        let norm_factor = 1.0 / magnitude;
        let perp = Plane::new(
            perp_x * norm_factor,
            perp_y * norm_factor,
            perp_z * norm_factor,
            0.0,
        );
        super::geo_standard_path::perpendicular_point(
            pm,
            n,
            &self.start_cutoff_plane,
            &self.end_cutoff_plane,
            &perp,
            x,
            y,
            z,
        )
        .ok_or(true)
    }

    fn path_center_distance(
        &self,
        pm: &PlanetModel,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        match self.the_point(pm, x, y, z) {
            Err(false) => style.compute_distance(&self.start, x, y, z),
            Err(true) => f64::INFINITY,
            Ok(p) => style.to_aggregation_form(style.compute_distance(&p, x, y, z)),
        }
    }

    fn nearest_path_distance(
        &self,
        pm: &PlanetModel,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        if !self.is_within_section(x, y, z) {
            return f64::INFINITY;
        }
        match self.the_point(pm, x, y, z) {
            Err(false) => style.to_aggregation_form(0.0),
            Err(true) => f64::INFINITY,
            Ok(p) => style.to_aggregation_form(style.compute_distance(&self.start, p.x, p.y, p.z)),
        }
    }

    fn path_distance(&self, pm: &PlanetModel, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if !self.is_within(x, y, z) {
            return f64::INFINITY;
        }
        match self.the_point(pm, x, y, z) {
            Err(false) => style.to_aggregation_form(style.compute_distance(&self.start, x, y, z)),
            Err(true) => f64::INFINITY,
            Ok(p) => style.aggregate_distances(&[
                style.to_aggregation_form(style.compute_distance(&p, x, y, z)),
                style.to_aggregation_form(style.compute_distance(&self.start, p.x, p.y, p.z)),
            ]),
        }
    }

    fn outside_distance(
        &self,
        pm: &PlanetModel,
        style: DistanceStyle,
        x: f64,
        y: f64,
        z: f64,
    ) -> f64 {
        let distance = style.compute_distance_to_plane(
            pm,
            &self.normalized_connecting_plane,
            x,
            y,
            z,
            &[&self.start_cutoff_plane, &self.end_cutoff_plane],
        );
        let start_distance = style.compute_distance(&self.start, x, y, z);
        let end_distance = style.compute_distance(&self.end, x, y, z);
        min(min(start_distance, end_distance), distance)
    }

    fn intersects(
        &self,
        pm: &PlanetModel,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        self.normalized_connecting_plane.intersects(
            pm,
            p,
            &self.connecting_plane_points,
            notable_points,
            bounds,
            &[&self.start_cutoff_plane, &self.end_cutoff_plane],
        )
    }

    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.normalized_connecting_plane,
            &self.connecting_plane_points,
            &[&self.start_cutoff_plane, &self.end_cutoff_plane],
        )
    }

    fn get_bounds(&self, pm: &PlanetModel, bounds: &mut dyn Bounds) {
        // We need to do all bounding planes as well as corner points
        base_get_bounds(&SegmentMembership(self), pm, bounds);
        bounds
            .add_point(&self.start)
            .add_point(&self.end)
            .add_plane(
                pm,
                &self.normalized_connecting_plane,
                &[&self.start_cutoff_plane, &self.end_cutoff_plane],
            );
    }
}

struct SegmentMembership<'a>(&'a PathSegment);

impl Membership for SegmentMembership<'_> {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.0.is_within(x, y, z)
    }
}

/// A zero-width path.
#[derive(Debug, Clone)]
pub struct GeoDegeneratePath {
    planet_model: Arc<PlanetModel>,
    points: Vec<GeoPoint>,
    end_points: Vec<SegmentEndpoint>,
    segments: Vec<PathSegment>,
    edge_points: Vec<GeoPoint>,
}

impl GeoDegeneratePath {
    /// `GeoDegeneratePath(planetModel, pathPoints)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        path_points: &[GeoPoint],
    ) -> Result<GeoDegeneratePath> {
        let points: Vec<GeoPoint> = path_points.to_vec();
        if points.is_empty() {
            return Err(illegal("Path must have at least one point"));
        }
        let mut end_points = Vec::with_capacity(points.len());
        let mut segments: Vec<PathSegment> = Vec::with_capacity(points.len());
        // First, build all segments.  We'll then go back and build
        // corresponding segment endpoints.
        let mut last_point: Option<&GeoPoint> = None;
        for end in &points {
            if let Some(lp) = last_point {
                let normalized_connecting_plane = Plane::from_vectors(lp, end)?;
                segments.push(PathSegment::new(
                    lp.clone(),
                    end.clone(),
                    normalized_connecting_plane,
                )?);
            }
            last_point = Some(end);
        }
        let mut edge_points = Vec::new();
        if segments.is_empty() {
            // Simple point
            let point = points[0].clone();
            edge_points = vec![point.clone()];
            end_points.push(SegmentEndpoint {
                point,
                cutoff_planes: Vec::new(),
            });
        } else {
            // Create segment endpoints.  Use an appropriate constructor for the
            // start and end of the path.
            for i in 0..segments.len() {
                let current = &segments[i];
                if i == 0 {
                    // Starting endpoint
                    end_points.push(SegmentEndpoint {
                        point: current.start.clone(),
                        cutoff_planes: vec![SidedPlane::opposite(&current.start_cutoff_plane)],
                    });
                    edge_points = vec![current.start.clone()];
                    continue;
                }
                end_points.push(SegmentEndpoint {
                    point: current.start.clone(),
                    cutoff_planes: vec![
                        SidedPlane::opposite(&segments[i - 1].end_cutoff_plane),
                        SidedPlane::opposite(&current.start_cutoff_plane),
                    ],
                });
            }
            // Do final endpoint
            let last = &segments[segments.len() - 1];
            end_points.push(SegmentEndpoint {
                point: last.end.clone(),
                cutoff_planes: vec![SidedPlane::opposite(&last.end_cutoff_plane)],
            });
        }
        Ok(GeoDegeneratePath {
            planet_model: planet_model.clone(),
            points,
            end_points,
            segments,
            edge_points,
        })
    }

    /// `GeoDegeneratePath(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoDegeneratePath> {
        let points = super::standard_objects::read_point_array(input)?;
        GeoDegeneratePath::new(planet_model, &points)
    }

    fn distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        // Algorithm: (1) If the point is within any of the segments along the
        // path, return that value. (2) If the point is within any of the
        // segment end circles along the path, return that value.
        let mut current_distance = 0.0;
        for segment in &self.segments {
            let distance = segment.path_distance(pm, style, x, y, z);
            if distance != f64::INFINITY {
                return style.from_aggregation_form(
                    style.aggregate_distances(&[current_distance, distance]),
                );
            }
            current_distance =
                style.aggregate_distances(&[current_distance, segment.full_path_distance(style)]);
        }
        // The end points, each at the sum of the segments before it (summed
        // in Java's order, from 0).
        for (k, endpoint) in self.end_points.iter().enumerate() {
            let distance = endpoint.path_distance(style, x, y, z);
            if distance != f64::INFINITY {
                let along = self.segments[..k.min(self.segments.len())]
                    .iter()
                    .fold(0.0, |acc, s| {
                        style.aggregate_distances(&[acc, s.full_path_distance(style)])
                    });
                return style.from_aggregation_form(style.aggregate_distances(&[along, distance]));
            }
        }
        f64::INFINITY
    }

    fn delta_distance(&self, _style: DistanceStyle, _x: f64, _y: f64, _z: f64) -> f64 {
        0.0
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
        let pm = &*self.planet_model;
        let mut min_distance = f64::INFINITY;
        for endpoint in &self.end_points {
            let new_distance = endpoint.outside_distance(style, x, y, z);
            if new_distance < min_distance {
                min_distance = new_distance;
            }
        }
        for segment in &self.segments {
            let new_distance = segment.outside_distance(pm, style, x, y, z);
            if new_distance < min_distance {
                min_distance = new_distance;
            }
        }
        style.from_aggregation_form(min_distance)
    }
}

impl SerializableObject for GeoDegeneratePath {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        super::standard_objects::write_point_array(out, &self.points);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(36)
    }
}

impl_planet_object!(GeoDegeneratePath);
impl_membership_shape!(GeoDegeneratePath);
impl_base_area!(GeoDegeneratePath);
impl_distance_shape!(GeoDegeneratePath);

impl Membership for GeoDegeneratePath {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.end_points.iter().any(|p| p.is_within(x, y, z))
            || self.segments.iter().any(|s| s.is_within(x, y, z))
    }
}

impl Bounded for GeoDegeneratePath {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        // For building bounds, order matters.  We want to traverse never more
        // than 180 degrees longitude at a pop or we risk having the
        // bounds object get itself inverted.  So do the edges first.
        for segment in &self.segments {
            segment.get_bounds(pm, bounds);
        }
        if self.end_points.len() == 1 {
            bounds.add_point(&self.end_points[0].point);
        }
    }
}

impl GeoShape for GeoDegeneratePath {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        plane: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        // We look for an intersection with any of the exterior edges of the
        // path. We also have to look for intersections with the cones
        // described by the endpoints. Return "true" if any such intersections
        // are found.
        if self.end_points.len() == 1 {
            return self.end_points[0].intersects(plane, bounds);
        }
        self.segments
            .iter()
            .any(|s| s.intersects(&self.planet_model, plane, notable_points, bounds))
    }
}

impl GeoAreaShape for GeoDegeneratePath {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        if self.end_points.len() == 1 {
            return geo_shape.is_within(&self.end_points[0].point);
        }
        self.segments.iter().any(|s| s.intersects_shape(geo_shape))
    }
}

impl GeoPath for GeoDegeneratePath {
    fn compute_nearest_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let mut current_distance = 0.0;
        let mut min_path_center_distance = f64::INFINITY;
        let mut best_distance = f64::INFINITY;
        let mut segment_index = 0;
        for endpoint in &self.end_points {
            let endpoint_path_center_distance = endpoint.path_center_distance(style, x, y, z);
            if endpoint_path_center_distance < min_path_center_distance {
                // Use this endpoint
                min_path_center_distance = endpoint_path_center_distance;
                best_distance = current_distance;
            }
            // Look at the following segment, if any
            if segment_index < self.segments.len() {
                let segment = &self.segments[segment_index];
                segment_index += 1;
                let segment_path_center_distance = segment.path_center_distance(pm, style, x, y, z);
                if segment_path_center_distance < min_path_center_distance {
                    min_path_center_distance = segment_path_center_distance;
                    best_distance = style.aggregate_distances(&[
                        current_distance,
                        segment.nearest_path_distance(pm, style, x, y, z),
                    ]);
                }
                current_distance = style
                    .aggregate_distances(&[current_distance, segment.full_path_distance(style)]);
            }
        }
        style.from_aggregation_form(best_distance)
    }

    fn compute_path_center_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        // Walk along path and keep track of the closest distance we find
        let mut closest_distance = f64::INFINITY;
        // Segments first
        for segment in &self.segments {
            let segment_distance = segment.path_center_distance(pm, style, x, y, z);
            if segment_distance < closest_distance {
                closest_distance = segment_distance;
            }
        }
        // Now, endpoints
        for endpoint in &self.end_points {
            let endpoint_distance = endpoint.path_center_distance(style, x, y, z);
            if endpoint_distance < closest_distance {
                closest_distance = endpoint_distance;
            }
        }
        closest_distance
    }
}
