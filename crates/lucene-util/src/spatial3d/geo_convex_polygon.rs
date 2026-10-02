//! `GeoConvexPolygon` and `GeoConcavePolygon`
//! (`org.apache.lucene.spatial3d.geom`): a polygon whose edges are all
//! within 180 degrees of each other, inside every edge plane (convex) or
//! outside at least one (concave), optionally with holes.
//!
//! The two Java classes are line for line the same but for which side of
//! each edge plane is inside; they share [`EdgePolygon`] here, with the
//! differences under `concave`.
//!
//! Java keys `prevBrotherMap`/`nextBrotherMap` by `SidedPlane.equals`, so
//! two edges with equal planes share one entry -- the one put last. The
//! lookups here reproduce that (the last edge equal to the key wins).

use super::prelude::*;
use super::serializable::{read_bit_set, write_bit_set};
use super::shape::GeoPolygon;
use super::standard_objects::{
    read_point_array, read_polygon_array, write_point_array, write_polygon_array,
};

/// The edge structure both polygon kinds share.
#[derive(Debug, Clone)]
pub(crate) struct EdgePolygon {
    pub(crate) planet_model: Arc<PlanetModel>,
    pub(crate) points: Vec<GeoPoint>,
    /// `isInternalEdges`, one flag per edge (missing flags are false).
    pub(crate) is_internal_edges: Vec<bool>,
    /// `holes`, `None` for no holes.
    pub(crate) holes: Option<Vec<Arc<dyn GeoPolygon>>>,
    pub(crate) edges: Vec<SidedPlane>,
    /// Concave only: `invertedEdges`.
    pub(crate) inverted_edges: Vec<SidedPlane>,
    pub(crate) start_bounds: Vec<SidedPlane>,
    pub(crate) end_bounds: Vec<SidedPlane>,
    pub(crate) notable_edge_points: Vec<[GeoPoint; 2]>,
    pub(crate) edge_points: Vec<GeoPoint>,
    /// `prevBrotherMap`/`nextBrotherMap`, by the index of the key's edge.
    prev_brother: Vec<usize>,
    next_brother: Vec<usize>,
    pub(crate) concave: bool,
}

impl std::fmt::Debug for dyn GeoPolygon {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GeoPolygon(class {:?})", self.class_code())
    }
}

/// Java's `List.toString()` of points.
pub(crate) fn points_string(points: &[GeoPoint]) -> String {
    let parts: Vec<String> = points.iter().map(|p| p.to_string()).collect();
    format!("[{}]", parts.join(", "))
}

/// "Constructed planes are all coplanar": every edge plane is one plane.
fn all_coplanar(points: &[GeoPoint]) -> Error {
    Error::IllegalArgument(format!(
        "Constructed planes are all coplanar: {}",
        points_string(points)
    ))
}

/// `SidedPlane.equals`.
fn sided_equals(a: &SidedPlane, b: &SidedPlane) -> bool {
    a.x == b.x
        && a.y == b.y
        && a.z == b.z
        && a.D == b.D
        && super::jmath::double_compare_eq(a.sig_num, b.sig_num)
}

impl EdgePolygon {
    /// The constructors' shared body: `done(isInternalReturnEdge)`.
    pub(crate) fn new(
        planet_model: &Arc<PlanetModel>,
        points: Vec<GeoPoint>,
        holes: Option<Vec<Arc<dyn GeoPolygon>>>,
        is_internal_edges: Vec<bool>,
        is_internal_return_edge: bool,
        concave: bool,
    ) -> Result<EdgePolygon> {
        let holes = holes.filter(|h| !h.is_empty());
        let mut is_internal_edges = is_internal_edges;
        let n = points.len();
        if n < 3 {
            return Err(illegal("Polygon needs at least three points."));
        }
        if is_internal_return_edge {
            if is_internal_edges.len() < n {
                is_internal_edges.resize(n, false);
            }
            is_internal_edges[n - 1] = true;
        }
        let legal_index = |index: isize| -> usize { index.rem_euclid(n as isize) as usize };
        let mut edges = Vec::with_capacity(n);
        let mut inverted_edges = Vec::new();
        let mut start_bounds = Vec::with_capacity(n);
        let mut end_bounds = Vec::with_capacity(n);
        let mut notable_edge_points = Vec::with_capacity(n);
        for i in 0..n {
            let start = &points[i];
            let end = &points[legal_index(i as isize + 1)];
            // We have to find the next point that is not on the plane between
            // start and end. If there is no such point, it's an error.
            let plane_to_find = Plane::from_vectors(start, end)?;
            let mut end_point_index = None;
            for j in 0..n {
                let index = legal_index((j + i) as isize + 2);
                if !plane_to_find.evaluate_is_zero(&points[index]) {
                    end_point_index = Some(index);
                    break;
                }
            }
            let Some(end_point_index) = end_point_index else {
                return Err(Error::IllegalArgument(if concave {
                    "Polygon points are all coplanar".to_string()
                } else {
                    format!(
                        "Polygon points are all coplanar: {}",
                        points_string(&points)
                    )
                }));
            };
            let check = &points[end_point_index];
            let sp = if concave {
                // Note: SidedPlane constructor here is "flipped"
                SidedPlane::from_vectors_on_side(check, false, start, end)?
            } else {
                SidedPlane::from_vectors(check, start, end)?
            };
            let start_bound = SidedPlane::construct_sided_plane_from_one_point(end, &sp, start);
            start_bounds.push(start_bound?);
            let end_bound = SidedPlane::construct_sided_plane_from_one_point(start, &sp, end);
            end_bounds.push(end_bound?);
            if concave {
                inverted_edges.push(SidedPlane::opposite(&sp));
            }
            edges.push(sp);
            notable_edge_points.push([start.clone(), end.clone()]);
        }
        // For each edge, create a bounds object.
        let mut prev_brother = vec![0; n];
        let mut next_brother = vec![0; n];
        {
            let planes = if concave { &inverted_edges } else { &edges };
            for edge_index in 0..n {
                let edge = &planes[edge_index];
                // Java's two loops, one forward (bound1) and one backward
                // (bound2): the next plane either way that is not the edge's.
                let find_bound = |step: isize| -> Result<usize> {
                    let mut index = legal_index(edge_index as isize + step);
                    while planes[index].is_numerically_identical_plane(edge) {
                        if index == edge_index {
                            return Err(all_coplanar(&points));
                        }
                        index = legal_index(index as isize + step);
                    }
                    Ok(index)
                };
                let bound1_index = find_bound(1)?;
                // Look for bound2
                let bound2_index = find_bound(-1)?;
                // Validate the bounds
                let mut starting_index = bound2_index;
                loop {
                    starting_index = legal_index(starting_index as isize + 1);
                    if starting_index == bound1_index {
                        break;
                    }
                    let interior_point = &points[starting_index];
                    if !planes[bound1_index].is_within(interior_point)
                        || !planes[bound2_index].is_within(interior_point)
                    {
                        return Err(illegal(if concave {
                            "Concave polygon has a side that is more than 180 degrees"
                        } else {
                            "Convex polygon has a side that is more than 180 degrees"
                        }));
                    }
                }
                next_brother[edge_index] = bound1_index;
                prev_brother[edge_index] = bound2_index;
            }
        }
        // Pick an edge point arbitrarily from the outer polygon.  Glom this
        // together with all edge points from inner polygons.
        let mut edge_points = vec![points[0].clone()];
        if let Some(hs) = &holes {
            for hole in hs {
                edge_points.extend(hole.edge_points().iter().cloned());
            }
        }
        let p = EdgePolygon {
            planet_model: planet_model.clone(),
            points,
            is_internal_edges,
            holes,
            edges,
            inverted_edges,
            start_bounds,
            end_bounds,
            notable_edge_points,
            edge_points,
            prev_brother,
            next_brother,
            concave,
        };
        if p.is_within_holes(&p.points[0]) {
            return Err(illegal(
                "Polygon edge intersects a polygon hole; not allowed",
            ));
        }
        Ok(p)
    }

    /// `isWithinHoles(point)`: outside some hole's polygon (that is, inside
    /// the hole).
    fn is_within_holes(&self, point: &GeoPoint) -> bool {
        self.holes
            .iter()
            .flatten()
            .any(|hole| !hole.is_within(point))
    }

    fn internal(&self, i: usize) -> bool {
        self.is_internal_edges.get(i).copied().unwrap_or(false)
    }

    /// The brother planes are keyed by `SidedPlane.equals`: the last edge
    /// equal to `edges[i]` holds the entry Java finds.
    fn brother(&self, map: &[usize], i: usize) -> usize {
        let planes = if self.concave {
            &self.inverted_edges
        } else {
            &self.edges
        };
        let key = &planes[i];
        let mut found = i;
        for (j, e) in planes.iter().enumerate() {
            if sided_equals(e, key) {
                found = j;
            }
        }
        map[found]
    }

    pub(crate) fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        if !self.local_is_within(x, y, z) {
            return false;
        }
        if let Some(hs) = &self.holes {
            for polygon in hs {
                if !polygon.is_within_xyz(x, y, z) {
                    return false;
                }
            }
        }
        true
    }

    fn local_is_within(&self, x: f64, y: f64, z: f64) -> bool {
        if self.concave {
            // If present within *any* plane, then it is a member, except where
            // there are holes.
            self.edges.iter().any(|e| e.is_within_xyz(x, y, z))
        } else {
            self.edges.iter().all(|e| e.is_within_xyz(x, y, z))
        }
    }

    fn local_is_within_v(&self, v: &Vector) -> bool {
        self.local_is_within(v.x, v.y, v.z)
    }

    pub(crate) fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        let planes = if self.concave {
            &self.inverted_edges
        } else {
            &self.edges
        };
        // Java's indexed loop over four parallel arrays.
        #[allow(clippy::needless_range_loop)]
        for edge_index in 0..planes.len() {
            if !self.internal(edge_index)
                && planes[edge_index].intersects(
                    &self.planet_model,
                    p,
                    notable_points,
                    &self.notable_edge_points[edge_index],
                    bounds,
                    &[&self.start_bounds[edge_index], &self.end_bounds[edge_index]],
                )
            {
                return true;
            }
        }
        if let Some(hs) = &self.holes {
            for hole in hs {
                if hole.intersects(p, notable_points, bounds) {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn intersects_shape(&self, shape: &dyn GeoShape) -> bool {
        for edge_index in 0..self.edges.len() {
            if !self.internal(edge_index)
                && shape.intersects(
                    &self.edges[edge_index],
                    &self.notable_edge_points[edge_index],
                    &[&self.start_bounds[edge_index], &self.end_bounds[edge_index]],
                )
            {
                return true;
            }
        }
        if let Some(hs) = &self.holes {
            for hole in hs {
                if hole.intersects_shape(shape) {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        // Because of holes, we don't want to use superclass method
        if self.local_is_within_v(&pm.north_pole) {
            bounds
                .no_top_latitude_bound()
                .no_longitude_bound()
                .add_point(&pm.north_pole);
        }
        if self.local_is_within_v(&pm.south_pole) {
            bounds
                .no_bottom_latitude_bound()
                .no_longitude_bound()
                .add_point(&pm.south_pole);
        }
        if self.local_is_within_v(&pm.min_x_pole) {
            bounds.add_point(&pm.min_x_pole);
        }
        if self.local_is_within_v(&pm.max_x_pole) {
            bounds.add_point(&pm.max_x_pole);
        }
        if self.local_is_within_v(&pm.min_y_pole) {
            bounds.add_point(&pm.min_y_pole);
        }
        if self.local_is_within_v(&pm.max_y_pole) {
            bounds.add_point(&pm.max_y_pole);
        }
        if self.concave {
            bounds.is_wide();
        }
        // Add all the points
        for point in &self.points {
            bounds.add_point(point);
        }
        // Add planes with membership.
        let intersection = |bounds: &mut dyn Bounds, i: usize| {
            let planes = if self.concave {
                &self.inverted_edges
            } else {
                &self.edges
            };
            let next = self.brother(&self.next_brother, i);
            let prev = self.brother(&self.prev_brother, i);
            let next_next = self.brother(&self.next_brother, next);
            bounds.add_intersection(
                pm,
                &planes[i],
                &planes[next],
                &[&planes[prev], &planes[next_next]],
            );
        };
        for edge_index in 0..self.edges.len() {
            bounds.add_plane(
                pm,
                &self.edges[edge_index],
                &[&self.start_bounds[edge_index], &self.end_bounds[edge_index]],
            );
            if !self.concave {
                intersection(bounds, edge_index);
            }
        }
        if self.concave {
            for edge_index in 0..self.inverted_edges.len() {
                intersection(bounds, edge_index);
            }
        }
    }

    pub(crate) fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let mut minimum_distance = f64::INFINITY;
        for edge_point in &self.points {
            let new_dist = style.compute_distance(edge_point, x, y, z);
            if new_dist < minimum_distance {
                minimum_distance = new_dist;
            }
        }
        for edge_index in 0..self.edges.len() {
            let new_dist = style.compute_distance_to_plane(
                &self.planet_model,
                &self.edges[edge_index],
                x,
                y,
                z,
                &[&self.start_bounds[edge_index], &self.end_bounds[edge_index]],
            );
            if new_dist < minimum_distance {
                minimum_distance = new_dist;
            }
        }
        if let Some(hs) = &self.holes {
            for hole in hs {
                let hole_distance = hole.compute_outside_distance(style, x, y, z);
                if hole_distance != 0.0 && hole_distance < minimum_distance {
                    minimum_distance = hole_distance;
                }
            }
        }
        minimum_distance
    }

    /// `write(OutputStream)`: points, holes, the internal-edge bit set.
    pub(crate) fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_point_array(out, &self.points);
        write_polygon_array(out, self.holes.as_deref().unwrap_or(&[]))?;
        write_bit_set(out, &self.is_internal_edges);
        Ok(())
    }

    /// The stream constructor's fields.
    pub(crate) fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
        concave: bool,
    ) -> Result<EdgePolygon> {
        let points = read_point_array(input)?;
        let holes = read_polygon_array(planet_model, input)?;
        let bits = read_bit_set(input)?;
        let n = points.len();
        let flags: Vec<bool> = (0..n.max(1)).map(|i| bits.get(i)).collect();
        // `done(isInternalEdges.get(points.size() - 1))`; Java's `get(-1)`
        // throws for an empty point list, which `new` rejects anyway.
        let return_internal = n > 0 && bits.get(n - 1);
        EdgePolygon::new(
            planet_model,
            points,
            Some(holes),
            flags,
            return_internal,
            concave,
        )
    }
}

macro_rules! edge_polygon_type {
    ($t:ident, $java:literal, $code:expr, $concave:expr) => {
        #[doc = concat!("`", $java, "`.")]
        #[derive(Debug, Clone)]
        pub struct $t {
            planet_model: Arc<PlanetModel>,
            inner: EdgePolygon,
        }

        impl $t {
            #[doc = concat!("`", $java, "(planetModel, pointList, holes, internalEdgeFlags, returnEdgeInternal)`: `holes` may be `None`; `internal_edge_flags` may be shorter than the points (missing flags are false).")]
            pub fn new(
                planet_model: &Arc<PlanetModel>,
                point_list: Vec<GeoPoint>,
                holes: Option<Vec<Arc<dyn GeoPolygon>>>,
                internal_edge_flags: Vec<bool>,
                return_edge_internal: bool,
            ) -> Result<$t> {
                Ok($t {
                    planet_model: planet_model.clone(),
                    inner: EdgePolygon::new(planet_model, point_list, holes, internal_edge_flags, return_edge_internal, $concave)?,
                })
            }

            #[doc = concat!("`", $java, "(planetModel, pointList, holes)`.")]
            pub fn with_holes(
                planet_model: &Arc<PlanetModel>,
                point_list: Vec<GeoPoint>,
                holes: Option<Vec<Arc<dyn GeoPolygon>>>,
            ) -> Result<$t> {
                $t::new(planet_model, point_list, holes, Vec::new(), false)
            }

            #[doc = concat!("`", $java, "(planetModel, InputStream)`.")]
            pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<$t> {
                Ok($t {
                    planet_model: planet_model.clone(),
                    inner: EdgePolygon::read(planet_model, input, $concave)?,
                })
            }

            fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
                self.inner.outside_distance(style, x, y, z)
            }
        }

        impl SerializableObject for $t {
            fn write(&self, out: &mut Vec<u8>) -> Result<()> {
                self.inner.write(out)
            }

            fn class_code(&self) -> Option<u8> {
                Some($code)
            }
        }

        impl_planet_object!($t);
        impl_membership_shape!($t);
        impl_base_area!($t);

        impl Membership for $t {
            fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
                self.inner.is_within_xyz(x, y, z)
            }
        }

        impl Bounded for $t {
            fn get_bounds(&self, bounds: &mut dyn Bounds) {
                self.inner.get_bounds(bounds)
            }
        }

        impl GeoShape for $t {
            fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
                Cow::Borrowed(&self.inner.edge_points)
            }

            fn intersects(&self, p: &Plane, notable_points: &[GeoPoint], bounds: &[&dyn Membership]) -> bool {
                self.inner.intersects(p, notable_points, bounds)
            }
        }

        impl GeoAreaShape for $t {
            fn intersects_shape(&self, shape: &dyn GeoShape) -> bool {
                self.inner.intersects_shape(shape)
            }
        }

        impl GeoPolygon for $t {}
    };
}

edge_polygon_type!(GeoConvexPolygon, "GeoConvexPolygon", 4, false);
edge_polygon_type!(GeoConcavePolygon, "GeoConcavePolygon", 5, true);
