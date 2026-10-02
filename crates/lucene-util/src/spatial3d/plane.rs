//! `Plane` (`org.apache.lucene.spatial3d.geom.Plane`): `Ax + By + Cz + D = 0`,
//! and everything geo3d computes from planes -- their intersections with
//! each other on the ellipsoid, distances to them, interpolation along them,
//! and the latitude/longitude and x/y/z extremes they reach within bounds.
//!
//! Java's `Plane extends Vector`: here the normal `(A, B, C)` is a
//! [`Vector`] the plane derefs to. Java's static overloads that differ only by
//! the declared type of the argument get distinct names
//! ([`Plane::is_numerically_identical_plane`] versus the inherited
//! [`Vector::is_numerically_identical`]). Methods that return `null` in Java
//! return `Option`.

#![allow(non_snake_case)]

use super::bounds::Bounds;
use super::geo_point::GeoPoint;
use super::jmath::{abs, atan2, cos, next_down, next_up, sin, sqrt};
use super::lat_lon_bounds::LatLonBounds;
use super::membership::{meets_all_bounds, Membership};
use super::planet_model::PlanetModel;
use super::vector::{
    Vector, MINIMUM_RESOLUTION, MINIMUM_RESOLUTION_CUBED, MINIMUM_RESOLUTION_SQUARED,
};
use super::xyz_bounds::XYZBounds;
use super::{Error, Result};

/// `Plane.NO_BOUNDS`.
pub const NO_BOUNDS: &[&dyn Membership] = &[];

/// A plane `x*A + y*B + z*C + D = 0`; `(A, B, C)` is the deref'd [`Vector`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Plane {
    /// The normal, `(A, B, C)`.
    pub v: Vector,
    /// `D`.
    pub D: f64,
}

impl std::ops::Deref for Plane {
    type Target = Vector;

    #[inline]
    fn deref(&self) -> &Vector {
        &self.v
    }
}

impl std::fmt::Display for Plane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::geo::java_double_string as d;
        write!(
            f,
            "[A={}, B={}; C={}; D={}]",
            d(self.x),
            d(self.y),
            d(self.z),
            d(self.D)
        )
    }
}

/// `Plane.normalYPlane`.
pub const NORMAL_Y_PLANE: Plane = Plane::new(0.0, 1.0, 0.0, 0.0);
/// `Plane.normalXPlane`.
pub const NORMAL_X_PLANE: Plane = Plane::new(1.0, 0.0, 0.0, 0.0);
/// `Plane.normalZPlane`.
pub const NORMAL_Z_PLANE: Plane = Plane::new(0.0, 0.0, 1.0, 0.0);

/// The point `(x0, y0, z0)` on both planes `p` and `q`, chosen to maximize
/// the determinant -- the shared first step of `findIntersections`,
/// `findCrossings`, `intersects` and `crosses`. `None` where Java returns
/// its "no points" answer.
#[inline]
fn line_origin(p: &Plane, q: &Plane) -> Option<(f64, f64, f64)> {
    let denomYZ = p.y * q.z - p.z * q.y;
    let denomXZ = p.x * q.z - p.z * q.x;
    let denomXY = p.x * q.y - p.y * q.x;
    if abs(denomYZ) >= abs(denomXZ) && abs(denomYZ) >= abs(denomXY) {
        // X is the biggest, so our point will have x0 = 0.0
        if abs(denomYZ) < MINIMUM_RESOLUTION_SQUARED {
            return None;
        }
        let denom = 1.0 / denomYZ;
        Some((
            0.0,
            (-p.D * q.z - p.z * -q.D) * denom,
            (p.y * -q.D + p.D * q.y) * denom,
        ))
    } else if abs(denomXZ) >= abs(denomXY) && abs(denomXZ) >= abs(denomYZ) {
        // Y is the biggest, so y0 = 0.0
        if abs(denomXZ) < MINIMUM_RESOLUTION_SQUARED {
            return None;
        }
        let denom = 1.0 / denomXZ;
        Some((
            (-p.D * q.z - p.z * -q.D) * denom,
            0.0,
            (p.x * -q.D + p.D * q.x) * denom,
        ))
    } else {
        // Z is the biggest, so Z0 = 0.0
        if abs(denomXY) < MINIMUM_RESOLUTION_SQUARED {
            return None;
        }
        let denom = 1.0 / denomXY;
        Some((
            (-p.D * q.y - p.y * -q.D) * denom,
            (p.x * -q.D + p.D * q.x) * denom,
            0.0,
        ))
    }
}

/// The quadratic `A t^2 + B t + C = 0` whose roots are the line
/// `lineVector * t + (x0, y0, z0)`'s intersections with the ellipsoid.
#[inline]
fn line_quadratic(
    pm: &PlanetModel,
    lvx: f64,
    lvy: f64,
    lvz: f64,
    x0: f64,
    y0: f64,
    z0: f64,
) -> (f64, f64, f64) {
    let A = lvx * lvx * pm.inverse_xy_scaling_squared
        + lvy * lvy * pm.inverse_xy_scaling_squared
        + lvz * lvz * pm.inverse_z_scaling_squared;
    let B = 2.0
        * (lvx * x0 * pm.inverse_xy_scaling_squared
            + lvy * y0 * pm.inverse_xy_scaling_squared
            + lvz * z0 * pm.inverse_z_scaling_squared);
    let C = x0 * x0 * pm.inverse_xy_scaling_squared
        + y0 * y0 * pm.inverse_xy_scaling_squared
        + z0 * z0 * pm.inverse_z_scaling_squared
        - 1.0;
    (A, B, C)
}

impl Plane {
    /// `Plane(A, B, C, D)`.
    #[inline]
    pub const fn new(A: f64, B: f64, C: f64, D: f64) -> Plane {
        Plane {
            v: Vector::new(A, B, C),
            D,
        }
    }

    /// `Plane(Vector A, BX, BY, BZ)`: through the origin and both vectors.
    pub fn from_vector_xyz(A: &Vector, BX: f64, BY: f64, BZ: f64) -> Result<Plane> {
        Ok(Plane {
            v: Vector::perpendicular_to(A, BX, BY, BZ)?,
            D: 0.0,
        })
    }

    /// `Plane(Vector A, Vector B)`: through the origin and both vectors.
    pub fn from_vectors(A: &Vector, B: &Vector) -> Result<Plane> {
        Ok(Plane {
            v: Vector::perpendicular(A, B)?,
            D: 0.0,
        })
    }

    /// `Plane(planetModel, sinLat)`: the horizontal plane at a latitude.
    pub fn horizontal(pm: &PlanetModel, sin_lat: f64) -> Plane {
        Plane::new(
            0.0,
            0.0,
            1.0,
            -sin_lat * Vector::compute_desired_ellipsoid_magnitude_z(pm, sin_lat),
        )
    }

    /// `Plane(x, y)`: the vertical plane through the z axis and `(x, y)`.
    pub fn vertical(x: f64, y: f64) -> Plane {
        Plane::new(y, -x, 0.0, 0.0)
    }

    /// `Plane(Vector v, D)`.
    pub fn with_d(v: &Vector, D: f64) -> Plane {
        Plane { v: *v, D }
    }

    /// `Plane(basePlane, above)`: parallel, a resolution step above or
    /// below.
    pub fn offset(base: &Plane, above: bool) -> Plane {
        Plane::new(
            base.x,
            base.y,
            base.z,
            if above {
                next_up(base.D + MINIMUM_RESOLUTION)
            } else {
                next_down(base.D - MINIMUM_RESOLUTION)
            },
        )
    }

    /// `constructPerpendicularCenterPlaneOnePoint(plane, M)`.
    pub fn construct_perpendicular_center_plane_one_point(
        plane: &Plane,
        M: &Vector,
    ) -> Result<Plane> {
        let A0 = plane.x;
        let B0 = plane.y;
        let C0 = plane.z;
        let a1Denom = C0 * M.y - B0 * M.z;
        let b1Denom = C0 * M.x - A0 * M.z;
        let c1Denom = B0 * M.x - A0 * M.y;
        let A1;
        let B1;
        let C1;
        if abs(a1Denom) >= abs(b1Denom) && abs(a1Denom) >= abs(c1Denom) {
            A1 = 1.0;
            if abs(M.y) >= abs(M.z) {
                C1 = (B0 * M.x - A0 * M.y) / a1Denom;
                B1 = (-M.x - C1 * M.z) / M.y;
            } else {
                B1 = (A0 * M.z - C0 * M.x) / a1Denom;
                C1 = (-M.x - B1 * M.y) / M.z;
            }
        } else if abs(b1Denom) >= abs(a1Denom) && abs(b1Denom) >= abs(c1Denom) {
            B1 = 1.0;
            if abs(M.x) >= abs(M.z) {
                C1 = (A0 * M.y - B0 * M.x) / b1Denom;
                A1 = (-M.y - C1 * M.z) / M.x;
            } else {
                A1 = (B0 * M.z - C0 * M.y) / b1Denom;
                C1 = (-M.y - A1 * M.x) / M.z;
            }
        } else if abs(c1Denom) >= abs(a1Denom) && abs(c1Denom) >= abs(b1Denom) {
            C1 = 1.0;
            if abs(M.x) >= abs(M.y) {
                B1 = (A0 * M.z - C0 * M.x) / c1Denom;
                A1 = (-M.z - B1 * M.y) / M.x;
            } else {
                A1 = (C0 * M.y - B0 * M.z) / c1Denom;
                B1 = (-M.z - A1 * M.x) / M.y;
            }
        } else {
            return Err(Error::IllegalArgument(
                "Cannot find perpendicular plane as requested".into(),
            ));
        }
        let norm_factor = 1.0 / sqrt(A1 * A1 + B1 * B1 + C1 * C1);
        let v = Vector::new(A1 * norm_factor, B1 * norm_factor, C1 * norm_factor);
        Ok(Plane::with_d(&v, -(v.x * M.x + v.y * M.y + v.z * M.z)))
    }

    /// `constructPerpendicularCenterPlaneTwoPoints(M, N)`.
    pub fn construct_perpendicular_center_plane_two_points(
        M: &Vector,
        N: &Vector,
    ) -> Result<Plane> {
        let center_plane = Plane::from_vectors(M, N)?;
        let A0 = center_plane.x;
        let B0 = center_plane.y;
        let C0 = center_plane.z;
        let xDiff = M.x - N.x;
        let yDiff = M.y - N.y;
        let zDiff = M.z - N.z;
        if xDiff * xDiff + yDiff * yDiff + zDiff * zDiff < MINIMUM_RESOLUTION_SQUARED {
            return Err(Error::IllegalArgument(
                "Chosen points are numerically identical".into(),
            ));
        }
        let A1;
        let B1;
        let C1;
        let A1choice = C0 * yDiff - B0 * zDiff;
        let B1choice = C0 * xDiff - A0 * zDiff;
        let C1choice = B0 * xDiff - A0 * yDiff;
        if abs(A1choice) >= abs(B1choice) && abs(A1choice) >= abs(C1choice) {
            A1 = 1.0;
            if abs(yDiff) >= abs(zDiff) {
                C1 = (B0 * xDiff - A0 * yDiff) / A1choice;
                B1 = (-C1 * zDiff - xDiff) / yDiff;
            } else {
                B1 = (A0 * zDiff - C0 * xDiff) / A1choice;
                C1 = (-B1 * yDiff - xDiff) / zDiff;
            }
        } else if abs(B1choice) >= abs(A1choice) && abs(B1choice) >= abs(C1choice) {
            B1 = 1.0;
            if abs(xDiff) >= abs(zDiff) {
                C1 = (A0 * yDiff - B0 * xDiff) / B1choice;
                A1 = (-C1 * zDiff - yDiff) / xDiff;
            } else {
                A1 = (B0 * zDiff - C0 * yDiff) / B1choice;
                C1 = (-A1 * xDiff - yDiff) / zDiff;
            }
        } else if abs(C1choice) >= abs(A1choice) && abs(C1choice) >= abs(B1choice) {
            C1 = 1.0;
            if abs(xDiff) >= abs(yDiff) {
                B1 = (A0 * zDiff - C0 * xDiff) / C1choice;
                A1 = (-B1 * yDiff - zDiff) / xDiff;
            } else {
                A1 = (C0 * yDiff - B0 * zDiff) / C1choice;
                B1 = (-A1 * xDiff - zDiff) / yDiff;
            }
        } else {
            return Err(Error::IllegalArgument(
                "Equation appears to be unsolveable".into(),
            ));
        }
        let norm_factor = 1.0 / sqrt(A1 * A1 + B1 * B1 + C1 * C1);
        let v = Vector::new(A1 * norm_factor, B1 * norm_factor, C1 * norm_factor);
        Ok(Plane::with_d(&v, -(v.x * M.x + v.y * M.y + v.z * M.z)))
    }

    /// `constructNormalizedZPlane(Vector...)`: through the point with the
    /// greatest x-y distance. `None` when every point is on the z axis.
    pub fn construct_normalized_z_plane_points(points: &[&Vector]) -> Option<Plane> {
        let mut best_distance = 0.0;
        let mut best: Option<&Vector> = None;
        for point in points {
            let d = point.x * point.x + point.y * point.y;
            if d > best_distance {
                best_distance = d;
                best = Some(point);
            }
        }
        // Java dereferences a null best point (NullPointerException).
        let b = best?;
        Plane::construct_normalized_z_plane(b.x, b.y)
    }

    /// `constructNormalizedYPlane(Vector...)`.
    pub fn construct_normalized_y_plane_points(points: &[&Vector]) -> Option<Plane> {
        let mut best_distance = 0.0;
        let mut best: Option<&Vector> = None;
        for point in points {
            let d = point.x * point.x + point.z * point.z;
            if d > best_distance {
                best_distance = d;
                best = Some(point);
            }
        }
        let b = best?;
        Plane::construct_normalized_y_plane(b.x, b.z, 0.0)
    }

    /// `constructNormalizedXPlane(Vector...)`.
    pub fn construct_normalized_x_plane_points(points: &[&Vector]) -> Option<Plane> {
        let mut best_distance = 0.0;
        let mut best: Option<&Vector> = None;
        for point in points {
            let d = point.y * point.y + point.z * point.z;
            if d > best_distance {
                best_distance = d;
                best = Some(point);
            }
        }
        let b = best?;
        Plane::construct_normalized_x_plane(b.y, b.z, 0.0)
    }

    /// `constructNormalizedZPlane(x, y)`.
    pub fn construct_normalized_z_plane(x: f64, y: f64) -> Option<Plane> {
        if abs(x) < MINIMUM_RESOLUTION && abs(y) < MINIMUM_RESOLUTION {
            return None;
        }
        let denom = 1.0 / sqrt(x * x + y * y);
        Some(Plane::new(y * denom, -x * denom, 0.0, 0.0))
    }

    /// `constructNormalizedYPlane(x, z, D)`.
    pub fn construct_normalized_y_plane(x: f64, z: f64, D: f64) -> Option<Plane> {
        if abs(x) < MINIMUM_RESOLUTION && abs(z) < MINIMUM_RESOLUTION {
            return None;
        }
        let denom = 1.0 / sqrt(x * x + z * z);
        Some(Plane::new(z * denom, 0.0, -x * denom, D))
    }

    /// `constructNormalizedXPlane(y, z, D)`.
    pub fn construct_normalized_x_plane(y: f64, z: f64, D: f64) -> Option<Plane> {
        if abs(y) < MINIMUM_RESOLUTION && abs(z) < MINIMUM_RESOLUTION {
            return None;
        }
        let denom = 1.0 / sqrt(y * y + z * z);
        Some(Plane::new(0.0, z * denom, -y * denom, D))
    }

    /// `evaluate(v)`.
    #[inline]
    pub fn evaluate(&self, v: &Vector) -> f64 {
        self.v.dot_product(v) + self.D
    }

    /// `evaluate(x, y, z)`.
    #[inline]
    pub fn evaluate_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        self.v.dot_product_xyz(x, y, z) + self.D
    }

    /// `evaluateIsZero(v)`.
    #[inline]
    pub fn evaluate_is_zero(&self, v: &Vector) -> bool {
        abs(self.evaluate(v)) < MINIMUM_RESOLUTION
    }

    /// `evaluateIsZero(x, y, z)`.
    #[inline]
    pub fn evaluate_is_zero_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        abs(self.evaluate_xyz(x, y, z)) < MINIMUM_RESOLUTION
    }

    /// `normalize()`: `None` for a degenerate normal.
    pub fn normalize(&self) -> Option<Plane> {
        self.v.normalize().map(|v| Plane::with_d(&v, self.D))
    }

    /// `arcDistance(planetModel, x, y, z, bounds)`: the arc distance from the
    /// point to the nearest point of the plane within the bounds.
    pub fn arc_distance(
        &self,
        pm: &PlanetModel,
        x: f64,
        y: f64,
        z: f64,
        bounds: &[&dyn Membership],
    ) -> f64 {
        if self.evaluate_is_zero_xyz(x, y, z) {
            if meets_all_bounds(x, y, z, bounds) {
                return 0.0;
            }
            return f64::INFINITY;
        }
        let perp_plane = Plane::new(
            self.y * z - self.z * y,
            self.z * x - self.x * z,
            self.x * y - self.y * x,
            0.0,
        );
        let intersection_points = self.find_intersections_two(pm, &perp_plane, NO_BOUNDS);
        let mut min_distance = f64::INFINITY;
        // Java iterates a null array (NullPointerException) when the planes
        // are numerically identical; that needs the point to be on this
        // plane, which returned above.
        for p in intersection_points.iter().flatten() {
            if meets_all_bounds(p.x, p.y, p.z, bounds) {
                let the_distance = p.arc_distance_xyz(x, y, z);
                if the_distance < min_distance {
                    min_distance = the_distance;
                }
            }
        }
        min_distance
    }

    /// `normalDistance(x, y, z, bounds)`.
    pub fn normal_distance(&self, x: f64, y: f64, z: f64, bounds: &[&dyn Membership]) -> f64 {
        let dist = self.evaluate_xyz(x, y, z);
        let perp_x = x - dist * self.x;
        let perp_y = y - dist * self.y;
        let perp_z = z - dist * self.z;
        if !meets_all_bounds(perp_x, perp_y, perp_z, bounds) {
            return f64::INFINITY;
        }
        abs(dist)
    }

    /// `normalDistanceSquared(x, y, z, bounds)`.
    pub fn normal_distance_squared(
        &self,
        x: f64,
        y: f64,
        z: f64,
        bounds: &[&dyn Membership],
    ) -> f64 {
        let normal = self.normal_distance(x, y, z, bounds);
        if normal == f64::INFINITY {
            return normal;
        }
        normal * normal
    }

    /// `linearDistance(planetModel, x, y, z, bounds)`.
    pub fn linear_distance(
        &self,
        pm: &PlanetModel,
        x: f64,
        y: f64,
        z: f64,
        bounds: &[&dyn Membership],
    ) -> f64 {
        if self.evaluate_is_zero_xyz(x, y, z) {
            if meets_all_bounds(x, y, z, bounds) {
                return 0.0;
            }
            return f64::INFINITY;
        }
        let perp_plane = Plane::new(
            self.y * z - self.z * y,
            self.z * x - self.x * z,
            self.x * y - self.y * x,
            0.0,
        );
        let intersection_points = self.find_intersections_two(pm, &perp_plane, NO_BOUNDS);
        let mut min_distance = f64::INFINITY;
        for p in intersection_points.iter().flatten() {
            if meets_all_bounds(p.x, p.y, p.z, bounds) {
                let the_distance = p.linear_distance_xyz(x, y, z);
                if the_distance < min_distance {
                    min_distance = the_distance;
                }
            }
        }
        min_distance
    }

    /// `linearDistanceSquared(planetModel, x, y, z, bounds)`.
    pub fn linear_distance_squared(
        &self,
        pm: &PlanetModel,
        x: f64,
        y: f64,
        z: f64,
        bounds: &[&dyn Membership],
    ) -> f64 {
        let linear_distance = self.linear_distance(pm, x, y, z, bounds);
        linear_distance * linear_distance
    }

    /// `interpolate(planetModel, start, end, proportions)`: points along the
    /// plane between `start` and `end` at the given proportions of the
    /// angle between them.
    pub fn interpolate(
        &self,
        pm: &PlanetModel,
        start: &GeoPoint,
        end: &GeoPoint,
        proportions: &[f64],
    ) -> Result<Vec<GeoPoint>> {
        let mut A = self.x;
        let mut B = self.y;
        let mut C = self.z;
        let transX = -self.D * A;
        let transY = -self.D * B;
        let transZ = -self.D * C;
        let cosRA;
        let sinRA;
        let cosHA;
        let sinHA;
        let magnitude = self.v.magnitude();
        if magnitude >= MINIMUM_RESOLUTION {
            let denom = 1.0 / magnitude;
            A *= denom;
            B *= denom;
            C *= denom;
            let xy_magnitude = sqrt(A * A + B * B);
            if xy_magnitude >= MINIMUM_RESOLUTION {
                let xy_denom = 1.0 / xy_magnitude;
                cosRA = A * xy_denom;
                sinRA = -B * xy_denom;
            } else {
                cosRA = 1.0;
                sinRA = 0.0;
            }
            sinHA = xy_magnitude;
            cosHA = C;
        } else {
            cosRA = 1.0;
            sinRA = 0.0;
            cosHA = 1.0;
            sinHA = 0.0;
        }
        let modified_start = modify(start, transX, transY, transZ, sinRA, cosRA, sinHA, cosHA);
        let modified_end = modify(end, transX, transY, transZ, sinRA, cosRA, sinHA, cosHA);
        if abs(modified_start.z) >= MINIMUM_RESOLUTION {
            return Err(Error::IllegalArgument(format!(
                "Start point was not on plane: {}",
                crate::geo::java_double_string(modified_start.z)
            )));
        }
        if abs(modified_end.z) >= MINIMUM_RESOLUTION {
            return Err(Error::IllegalArgument(format!(
                "End point was not on plane: {}",
                crate::geo::java_double_string(modified_end.z)
            )));
        }
        let start_angle = atan2(modified_start.y, modified_start.x);
        let end_angle = atan2(modified_end.y, modified_end.x);
        let start_magnitude =
            sqrt(modified_start.x * modified_start.x + modified_start.y * modified_start.y);
        // Java's declare-then-assign, kept.
        #[allow(clippy::needless_late_init)]
        let delta;
        let mut new_end_angle = end_angle;
        use std::f64::consts::PI;
        while new_end_angle < start_angle {
            new_end_angle += PI * 2.0;
        }
        if new_end_angle - start_angle <= PI {
            delta = new_end_angle - start_angle;
        } else {
            let mut new_start_angle = start_angle;
            while new_start_angle < end_angle {
                new_start_angle += PI * 2.0;
            }
            delta = new_start_angle - end_angle;
        }
        Ok(proportions
            .iter()
            .map(|&p| {
                let new_angle = start_angle + p * delta;
                let sin_new_angle = sin(new_angle);
                let cos_new_angle = cos(new_angle);
                let new_vector = Vector::new(
                    cos_new_angle * start_magnitude,
                    sin_new_angle * start_magnitude,
                    0.0,
                );
                reverse_modify(
                    pm,
                    &new_vector,
                    transX,
                    transY,
                    transZ,
                    sinRA,
                    cosRA,
                    sinHA,
                    cosHA,
                )
            })
            .collect())
    }

    /// `findIntersections(planetModel, q, bounds)`: the surface points on both
    /// planes within the bounds. `None` (Java's `null`) when the planes are
    /// numerically identical.
    pub fn find_intersections(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
    ) -> Option<Vec<GeoPoint>> {
        if self.is_numerically_identical_plane(q) {
            return None;
        }
        Some(self.find_intersections_more(pm, q, bounds, NO_BOUNDS))
    }

    /// `findCrossings(planetModel, q, bounds)`: as
    /// [`Self::find_intersections`], but a tangent point does not count.
    pub fn find_crossings(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
    ) -> Option<Vec<GeoPoint>> {
        if self.is_numerically_identical_plane(q) {
            return None;
        }
        Some(self.find_crossings_more(pm, q, bounds, NO_BOUNDS))
    }

    /// `arePointsCoplanar(A, B, C)`.
    pub fn are_points_coplanar(A: &Vector, B: &Vector, C: &Vector) -> Result<bool> {
        Ok(Vector::cross_product_evaluate_is_zero(A, B, C)?
            || Vector::cross_product_evaluate_is_zero(A, C, B)?
            || Vector::cross_product_evaluate_is_zero(B, C, A)?)
    }

    /// The protected `findIntersections(planetModel, q, bounds, moreBounds)`.
    pub fn find_intersections_more(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> Vec<GeoPoint> {
        self.intersections(pm, q, bounds, more_bounds)
            .into_iter()
            .flatten()
            .collect()
    }

    /// [`Self::find_intersections`] for the distance computations: at most
    /// two points, without allocating (Java returns an array; the hot
    /// distance loops here would otherwise allocate per call).
    #[inline]
    pub(crate) fn find_intersections_two(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
    ) -> [Option<GeoPoint>; 2] {
        self.find_intersections_arr(pm, q, bounds)
            .unwrap_or([None, None])
    }

    /// [`Self::find_intersections`] without allocating: `None` for a
    /// numerically identical plane (Java's `null`), else the points.
    #[inline]
    pub(crate) fn find_intersections_arr(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
    ) -> Option<[Option<GeoPoint>; 2]> {
        if self.is_numerically_identical_plane(q) {
            return None;
        }
        Some(self.intersections(pm, q, bounds, NO_BOUNDS))
    }

    /// The body of [`Self::find_intersections_more`]: the (up to two)
    /// points, in Java's order.
    fn intersections(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> [Option<GeoPoint>; 2] {
        let lvx = self.y * q.z - self.z * q.y;
        let lvy = self.z * q.x - self.x * q.z;
        let lvz = self.x * q.y - self.y * q.x;
        if abs(lvx) < MINIMUM_RESOLUTION
            && abs(lvy) < MINIMUM_RESOLUTION
            && abs(lvz) < MINIMUM_RESOLUTION
        {
            // Degenerate case: parallel planes
            return [None, None];
        }
        let Some((x0, y0, z0)) = line_origin(self, q) else {
            return [None, None];
        };
        let (A, B, C) = line_quadratic(pm, lvx, lvy, lvz, x0, y0, z0);
        let BsquaredMinus = B * B - 4.0 * A * C;
        if abs(BsquaredMinus) < MINIMUM_RESOLUTION_SQUARED {
            let inverse2A = 1.0 / (2.0 * A);
            // One solution only
            let t = -B * inverse2A;
            let px = lvx * t + x0;
            let py = lvy * t + y0;
            let pz = lvz * t + z0;
            if !meets_all_bounds(px, py, pz, bounds) || !meets_all_bounds(px, py, pz, more_bounds) {
                return [None, None];
            }
            [Some(GeoPoint::new(px, py, pz)), None]
        } else if BsquaredMinus > 0.0 {
            let inverse2A = 1.0 / (2.0 * A);
            let sqrt_term = sqrt(BsquaredMinus);
            let t1 = (-B + sqrt_term) * inverse2A;
            let t2 = (-B - sqrt_term) * inverse2A;
            two_points(lvx, lvy, lvz, x0, y0, z0, t1, t2, bounds, more_bounds)
        } else {
            [None, None]
        }
    }

    /// The protected `findCrossings(planetModel, q, bounds, moreBounds)`.
    pub fn find_crossings_more(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> Vec<GeoPoint> {
        let lvx = self.y * q.z - self.z * q.y;
        let lvy = self.z * q.x - self.x * q.z;
        let lvz = self.x * q.y - self.y * q.x;
        if abs(lvx) < MINIMUM_RESOLUTION
            && abs(lvy) < MINIMUM_RESOLUTION
            && abs(lvz) < MINIMUM_RESOLUTION
        {
            return Vec::new();
        }
        let Some((x0, y0, z0)) = line_origin(self, q) else {
            return Vec::new();
        };
        let (A, B, C) = line_quadratic(pm, lvx, lvy, lvz, x0, y0, z0);
        let BsquaredMinus = B * B - 4.0 * A * C;
        if abs(BsquaredMinus) < MINIMUM_RESOLUTION_SQUARED {
            Vec::new()
        } else if BsquaredMinus > 0.0 {
            let inverse2A = 1.0 / (2.0 * A);
            let sqrt_term = sqrt(BsquaredMinus);
            let t1 = (-B + sqrt_term) * inverse2A;
            let t2 = (-B - sqrt_term) * inverse2A;
            two_points(lvx, lvy, lvz, x0, y0, z0, t1, t2, bounds, more_bounds)
                .into_iter()
                .flatten()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// `findIntersectionBounds(planetModel, boundsInfo, q, bounds)`: records
    /// the bounds of the intersection of this plane and `q`, each moved a
    /// resolution step either way.
    pub fn find_intersection_bounds(
        &self,
        pm: &PlanetModel,
        bounds_info: &mut dyn Bounds,
        q: &Plane,
        bounds: &[&dyn Membership],
    ) {
        let lvx = self.y * q.z - self.z * q.y;
        let lvy = self.z * q.x - self.x * q.z;
        let lvz = self.x * q.y - self.y * q.x;
        if abs(lvx) < MINIMUM_RESOLUTION
            && abs(lvy) < MINIMUM_RESOLUTION
            && abs(lvz) < MINIMUM_RESOLUTION
        {
            return;
        }
        let denomYZ = self.y * q.z - self.z * q.y;
        let denomXZ = self.x * q.z - self.z * q.x;
        let denomXY = self.x * q.y - self.y * q.x;
        let r = MINIMUM_RESOLUTION;
        let (D, qD) = (self.D, q.D);
        // The four combinations of `this.D +- r` and `q.D +- r`, in Java's
        // order: (+,+), (-,+), (+,-), (-,-).
        let combos = [
            (D + r, qD + r),
            (D - r, qD + r),
            (D + r, qD - r),
            (D - r, qD - r),
        ];
        if abs(denomYZ) >= abs(denomXZ) && abs(denomYZ) >= abs(denomXY) {
            if abs(denomYZ) < MINIMUM_RESOLUTION_SQUARED {
                return;
            }
            let denom = 1.0 / denomYZ;
            for (pD, qD) in combos {
                record_line_bounds(
                    pm,
                    bounds_info,
                    lvx,
                    lvy,
                    lvz,
                    0.0,
                    (-pD * q.z - self.z * -qD) * denom,
                    (self.y * -qD + pD * q.y) * denom,
                    bounds,
                );
            }
        } else if abs(denomXZ) >= abs(denomXY) && abs(denomXZ) >= abs(denomYZ) {
            if abs(denomXZ) < MINIMUM_RESOLUTION_SQUARED {
                return;
            }
            let denom = 1.0 / denomXZ;
            for (pD, qD) in combos {
                record_line_bounds(
                    pm,
                    bounds_info,
                    lvx,
                    lvy,
                    lvz,
                    (-pD * q.z - self.z * -qD) * denom,
                    0.0,
                    (self.x * -qD + pD * q.x) * denom,
                    bounds,
                );
            }
        } else {
            if abs(denomXY) < MINIMUM_RESOLUTION_SQUARED {
                return;
            }
            let denom = 1.0 / denomXY;
            for (pD, qD) in combos {
                record_line_bounds(
                    pm,
                    bounds_info,
                    lvx,
                    lvy,
                    lvz,
                    (-pD * q.y - self.y * -qD) * denom,
                    (self.x * -qD + pD * q.x) * denom,
                    0.0,
                    bounds,
                );
            }
        }
    }

    /// `recordBounds(planetModel, XYZBounds, p, bounds)`.
    pub fn record_bounds_xyz_intersection(
        &self,
        pm: &PlanetModel,
        bounds_info: &mut XYZBounds,
        p: &Plane,
        bounds: &[&dyn Membership],
    ) {
        self.find_intersection_bounds(pm, bounds_info, p, bounds);
    }

    /// `recordBounds(planetModel, XYZBounds, bounds)`: the x, y and z extremes
    /// of the plane's intersection with the ellipsoid, within the bounds.
    pub fn record_bounds_xyz(
        &self,
        pm: &PlanetModel,
        bounds_info: &mut XYZBounds,
        bounds: &[&dyn Membership],
    ) {
        let A = self.x;
        let B = self.y;
        let C = self.z;
        let D = self.D;
        // Do Z.  This can be done simply because it is symmetrical.
        if !bounds_info.is_smallest_min_z(pm) || !bounds_info.is_largest_max_z(pm) {
            if abs(A) >= MINIMUM_RESOLUTION || abs(B) >= MINIMUM_RESOLUTION {
                // We need unconstrained values in order to compute D
                let normalized_z_plane =
                    Plane::construct_normalized_z_plane(A, B).expect("A or B is non-zero");
                let points =
                    self.find_intersections_more(pm, &normalized_z_plane, bounds, NO_BOUNDS);
                for point in &points {
                    add_point(bounds_info, bounds, point);
                }
            } else {
                // Since a==b==0, any plane including the Z axis suffices.
                let mut points =
                    self.find_intersections_more(pm, &NORMAL_Y_PLANE, NO_BOUNDS, NO_BOUNDS);
                if points.is_empty() {
                    points =
                        self.find_intersections_more(pm, &NORMAL_X_PLANE, NO_BOUNDS, NO_BOUNDS);
                }
                if points.is_empty() {
                    bounds_info.add_z_value(&GeoPoint::new(0.0, 0.0, -self.z));
                } else {
                    bounds_info.add_z_value(&points[0]);
                }
            }
        }
        let k = 1.0
            / ((self.x * self.x + self.y * self.y) * pm.xy_scaling * pm.xy_scaling
                + self.z * self.z * pm.z_scaling * pm.z_scaling);
        let abSquared = pm.xy_scaling * pm.xy_scaling;
        let cSquared = pm.z_scaling * pm.z_scaling;
        let ASquared = A * A;
        let BSquared = B * B;
        let CSquared = C * C;
        let r = 2.0 * D * k;
        let rSquared = r * r;
        if !bounds_info.is_smallest_min_x(pm) || !bounds_info.is_largest_max_x(pm) {
            let q = A * abSquared * k;
            let qSquared = q * q;
            let a = ASquared * abSquared * rSquared
                + BSquared * abSquared * rSquared
                + CSquared * cSquared * rSquared
                - 4.0;
            let b = -2.0 * A * abSquared * r
                + 2.0 * ASquared * abSquared * r * q
                + 2.0 * BSquared * abSquared * r * q
                + 2.0 * CSquared * cSquared * r * q;
            let c = abSquared - 2.0 * A * abSquared * q
                + ASquared * abSquared * qSquared
                + BSquared * abSquared * qSquared
                + CSquared * cSquared * qSquared;
            let point_x = |l: f64, denom: f64| {
                GeoPoint::new(
                    (1.0 - l * A) * abSquared * denom,
                    -l * B * abSquared * denom,
                    -l * C * cSquared * denom,
                )
            };
            if abs(a) >= MINIMUM_RESOLUTION_SQUARED {
                let sqrt_term = b * b - 4.0 * a * c;
                if abs(sqrt_term) < MINIMUM_RESOLUTION_SQUARED {
                    // One solution
                    let m = -b / (2.0 * a);
                    if abs(m) >= MINIMUM_RESOLUTION {
                        let l = r * m + q;
                        let denom0 = 0.5 / m;
                        add_point(bounds_info, bounds, &point_x(l, denom0));
                    } else {
                        bounds_info.add_x_value_f64(-D / A);
                    }
                } else if sqrt_term > 0.0 {
                    // Two solutions
                    let sqrt_result = sqrt(sqrt_term);
                    let common_denom = 0.5 / a;
                    let m1 = (-b + sqrt_result) * common_denom;
                    let m2 = (-b - sqrt_result) * common_denom;
                    if abs(m1) >= MINIMUM_RESOLUTION || abs(m2) >= MINIMUM_RESOLUTION {
                        let l1 = r * m1 + q;
                        let l2 = r * m2 + q;
                        let denom1 = 0.5 / m1;
                        let denom2 = 0.5 / m2;
                        let p1 = point_x(l1, denom1);
                        let p2 = point_x(l2, denom2);
                        add_point(bounds_info, bounds, &p1);
                        add_point(bounds_info, bounds, &p2);
                    } else {
                        bounds_info.add_x_value_f64(-D / A);
                    }
                }
            } else if abs(b) > MINIMUM_RESOLUTION_SQUARED {
                // a = 0, so m = - c / b
                let m = -c / b;
                let l = r * m + q;
                let denom0 = 0.5 / m;
                add_point(bounds_info, bounds, &point_x(l, denom0));
            }
        }
        // Do Y
        if !bounds_info.is_smallest_min_y(pm) || !bounds_info.is_largest_max_y(pm) {
            let q = B * abSquared * k;
            let qSquared = q * q;
            let a = ASquared * abSquared * rSquared
                + BSquared * abSquared * rSquared
                + CSquared * cSquared * rSquared
                - 4.0;
            let b = 2.0 * ASquared * abSquared * r * q - 2.0 * B * abSquared * r
                + 2.0 * BSquared * abSquared * r * q
                + 2.0 * CSquared * cSquared * r * q;
            let c = ASquared * abSquared * qSquared + abSquared - 2.0 * B * abSquared * q
                + BSquared * abSquared * qSquared
                + CSquared * cSquared * qSquared;
            let point_y = |l: f64, denom: f64| {
                GeoPoint::new(
                    -l * A * abSquared * denom,
                    (1.0 - l * B) * abSquared * denom,
                    -l * C * cSquared * denom,
                )
            };
            if abs(a) >= MINIMUM_RESOLUTION_SQUARED {
                let sqrt_term = b * b - 4.0 * a * c;
                if abs(sqrt_term) < MINIMUM_RESOLUTION_SQUARED {
                    let m = -b / (2.0 * a);
                    if abs(m) >= MINIMUM_RESOLUTION {
                        let l = r * m + q;
                        let denom0 = 0.5 / m;
                        add_point(bounds_info, bounds, &point_y(l, denom0));
                    } else {
                        bounds_info.add_y_value_f64(-D / B);
                    }
                } else if sqrt_term > 0.0 {
                    let sqrt_result = sqrt(sqrt_term);
                    let common_denom = 0.5 / a;
                    let m1 = (-b + sqrt_result) * common_denom;
                    let m2 = (-b - sqrt_result) * common_denom;
                    if abs(m1) >= MINIMUM_RESOLUTION || abs(m2) >= MINIMUM_RESOLUTION {
                        let l1 = r * m1 + q;
                        let l2 = r * m2 + q;
                        let denom1 = 0.5 / m1;
                        let denom2 = 0.5 / m2;
                        let p1 = point_y(l1, denom1);
                        let p2 = point_y(l2, denom2);
                        add_point(bounds_info, bounds, &p1);
                        add_point(bounds_info, bounds, &p2);
                    } else {
                        bounds_info.add_y_value_f64(-D / B);
                    }
                }
            } else if abs(b) > MINIMUM_RESOLUTION_SQUARED {
                let m = -c / b;
                let l = r * m + q;
                let denom0 = 0.5 / m;
                add_point(bounds_info, bounds, &point_y(l, denom0));
            }
        }
    }

    /// `recordBounds(planetModel, LatLonBounds, p, bounds)`.
    pub fn record_bounds_lat_lon_intersection(
        &self,
        pm: &PlanetModel,
        bounds_info: &mut LatLonBounds,
        p: &Plane,
        bounds: &[&dyn Membership],
    ) {
        self.find_intersection_bounds(pm, bounds_info, p, bounds);
    }

    /// `recordBounds(planetModel, LatLonBounds, bounds)`: the latitude and
    /// longitude extremes of the plane's intersection with the ellipsoid.
    pub fn record_bounds_lat_lon(
        &self,
        pm: &PlanetModel,
        bounds_info: &mut LatLonBounds,
        bounds: &[&dyn Membership],
    ) {
        let A = self.x;
        let B = self.y;
        let C = self.z;
        let D = self.D;
        // Look for the min and max latitude, if needed.
        if !bounds_info.check_no_top_latitude_bound()
            || !bounds_info.check_no_bottom_latitude_bound()
        {
            if abs(A) >= MINIMUM_RESOLUTION || abs(B) >= MINIMUM_RESOLUTION {
                let vertical_plane =
                    Plane::construct_normalized_z_plane(A, B).expect("A or B is non-zero");
                let points = self.find_intersections_more(pm, &vertical_plane, bounds, NO_BOUNDS);
                for point in &points {
                    add_point(bounds_info, bounds, point);
                }
            } else {
                let mut points =
                    self.find_intersections_more(pm, &NORMAL_X_PLANE, NO_BOUNDS, NO_BOUNDS);
                if points.is_empty() {
                    points =
                        self.find_intersections_more(pm, &NORMAL_Y_PLANE, NO_BOUNDS, NO_BOUNDS);
                }
                if points.is_empty() {
                    bounds_info.add_z_value(&GeoPoint::new(0.0, 0.0, -self.z));
                } else {
                    bounds_info.add_z_value(&points[0]);
                }
            }
        }
        // First, figure out our longitude bounds, unless we no longer need to
        // consider that
        if !bounds_info.check_no_longitude_bound() {
            let ixy = pm.inverse_xy_scaling_squared;
            let iz = pm.inverse_z_scaling_squared;
            if abs(C) < MINIMUM_RESOLUTION {
                // Degenerate; the equation describes a line
                if abs(D) >= MINIMUM_RESOLUTION {
                    if abs(A) > abs(B) {
                        let a = B * B * ixy + A * A * ixy;
                        let b = 2.0 * B * D * ixy;
                        let c = D * D * ixy - A * A;
                        let sqrt_clause = b * b - 4.0 * a * c;
                        if abs(sqrt_clause) < MINIMUM_RESOLUTION_SQUARED {
                            let y0 = -b / (2.0 * a);
                            let x0 = (-D - B * y0) / A;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0, y0, 0.0));
                        } else if sqrt_clause > 0.0 {
                            let sqrt_result = sqrt(sqrt_clause);
                            let denom = 1.0 / (2.0 * a);
                            let Hdenom = 1.0 / A;
                            let y0a = (-b + sqrt_result) * denom;
                            let y0b = (-b - sqrt_result) * denom;
                            let x0a = (-D - B * y0a) * Hdenom;
                            let x0b = (-D - B * y0b) * Hdenom;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0a, y0a, 0.0));
                            add_point(bounds_info, bounds, &GeoPoint::new(x0b, y0b, 0.0));
                        }
                    } else {
                        let a = B * B * ixy + A * A * ixy;
                        let b = 2.0 * A * D * ixy;
                        let c = D * D * ixy - B * B;
                        let sqrt_clause = b * b - 4.0 * a * c;
                        if abs(sqrt_clause) < MINIMUM_RESOLUTION_SQUARED {
                            let x0 = -b / (2.0 * a);
                            let y0 = (-D - A * x0) / B;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0, y0, 0.0));
                        } else if sqrt_clause > 0.0 {
                            let sqrt_result = sqrt(sqrt_clause);
                            let denom = 1.0 / (2.0 * a);
                            let Idenom = 1.0 / B;
                            let x0a = (-b + sqrt_result) * denom;
                            let x0b = (-b - sqrt_result) * denom;
                            let y0a = (-D - A * x0a) * Idenom;
                            let y0b = (-D - A * x0b) * Idenom;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0a, y0a, 0.0));
                            add_point(bounds_info, bounds, &GeoPoint::new(x0b, y0b, 0.0));
                        }
                    }
                }
            } else {
                let E = A * A * iz + C * C * ixy;
                let F = B * B * iz + C * C * ixy;
                let G = 2.0 * A * B * iz;
                let H = 2.0 * A * D * iz;
                let I = 2.0 * B * D * iz;
                let J = D * D * iz - C * C;
                if abs(J) >= MINIMUM_RESOLUTION && J > 0.0 {
                    if abs(H) > abs(I) {
                        let a = E * I * I - G * H * I + F * H * H;
                        let b = 4.0 * E * I * J - 2.0 * G * H * J;
                        let c = 4.0 * E * J * J - J * H * H;
                        let sqrt_clause = b * b - 4.0 * a * c;
                        if abs(sqrt_clause) < MINIMUM_RESOLUTION_CUBED {
                            let y0 = -b / (2.0 * a);
                            let x0 = (-2.0 * J - I * y0) / H;
                            let z0 = (-A * x0 - B * y0 - D) / C;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0, y0, z0));
                        } else if sqrt_clause > 0.0 {
                            let sqrt_result = sqrt(sqrt_clause);
                            let denom = 1.0 / (2.0 * a);
                            let Hdenom = 1.0 / H;
                            let Cdenom = 1.0 / C;
                            let y0a = (-b + sqrt_result) * denom;
                            let y0b = (-b - sqrt_result) * denom;
                            let x0a = (-2.0 * J - I * y0a) * Hdenom;
                            let x0b = (-2.0 * J - I * y0b) * Hdenom;
                            let z0a = (-A * x0a - B * y0a - D) * Cdenom;
                            let z0b = (-A * x0b - B * y0b - D) * Cdenom;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0a, y0a, z0a));
                            add_point(bounds_info, bounds, &GeoPoint::new(x0b, y0b, z0b));
                        }
                    } else {
                        let a = E * I * I - G * H * I + F * H * H;
                        let b = 4.0 * F * H * J - 2.0 * G * I * J;
                        let c = 4.0 * F * J * J - J * I * I;
                        let sqrt_clause = b * b - 4.0 * a * c;
                        if abs(sqrt_clause) < MINIMUM_RESOLUTION_CUBED {
                            let x0 = -b / (2.0 * a);
                            let y0 = (-2.0 * J - H * x0) / I;
                            let z0 = (-A * x0 - B * y0 - D) / C;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0, y0, z0));
                        } else if sqrt_clause > 0.0 {
                            let sqrt_result = sqrt(sqrt_clause);
                            let denom = 1.0 / (2.0 * a);
                            let Idenom = 1.0 / I;
                            let Cdenom = 1.0 / C;
                            let x0a = (-b + sqrt_result) * denom;
                            let x0b = (-b - sqrt_result) * denom;
                            let y0a = (-2.0 * J - H * x0a) * Idenom;
                            let y0b = (-2.0 * J - H * x0b) * Idenom;
                            let z0a = (-A * x0a - B * y0a - D) * Cdenom;
                            let z0b = (-A * x0b - B * y0b - D) * Cdenom;
                            add_point(bounds_info, bounds, &GeoPoint::new(x0a, y0a, z0a));
                            add_point(bounds_info, bounds, &GeoPoint::new(x0b, y0b, z0b));
                        }
                    }
                }
            }
        }
    }

    /// `intersects(planetModel, q, notablePoints, moreNotablePoints, bounds,
    /// moreBounds)`: whether the two planes meet on the surface within all
    /// bounds.
    pub fn intersects(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        notable_points: &[GeoPoint],
        more_notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> bool {
        self.intersects_or_crosses(
            pm,
            q,
            notable_points,
            more_notable_points,
            bounds,
            more_bounds,
            true,
        )
    }

    /// `crosses(...)`: as [`Self::intersects`], but a tangent point does not
    /// count.
    pub fn crosses(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        notable_points: &[GeoPoint],
        more_notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> bool {
        self.intersects_or_crosses(
            pm,
            q,
            notable_points,
            more_notable_points,
            bounds,
            more_bounds,
            false,
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    fn intersects_or_crosses(
        &self,
        pm: &PlanetModel,
        q: &Plane,
        notable_points: &[GeoPoint],
        more_notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
        tangent_counts: bool,
    ) -> bool {
        if self.is_numerically_identical_plane(q) {
            // The planes are identical.  We need to determine if *any* points
            // of the plane are within bounds
            return notable_points.iter().chain(more_notable_points).any(|p| {
                meets_all_bounds(p.x, p.y, p.z, bounds)
                    && meets_all_bounds(p.x, p.y, p.z, more_bounds)
            });
        }
        let lvx = self.y * q.z - self.z * q.y;
        let lvy = self.z * q.x - self.x * q.z;
        let lvz = self.x * q.y - self.y * q.x;
        if abs(lvx) < MINIMUM_RESOLUTION
            && abs(lvy) < MINIMUM_RESOLUTION
            && abs(lvz) < MINIMUM_RESOLUTION
        {
            return false;
        }
        let Some((x0, y0, z0)) = line_origin(self, q) else {
            return false;
        };
        let (A, B, C) = line_quadratic(pm, lvx, lvy, lvz, x0, y0, z0);
        let BsquaredMinus = B * B - 4.0 * A * C;
        if abs(BsquaredMinus) < MINIMUM_RESOLUTION_SQUARED {
            if !tangent_counts {
                return false;
            }
            let inverse2A = 1.0 / (2.0 * A);
            let t = -B * inverse2A;
            let px = lvx * t + x0;
            let py = lvy * t + y0;
            let pz = lvz * t + z0;
            meets_all_bounds(px, py, pz, bounds) && meets_all_bounds(px, py, pz, more_bounds)
        } else if BsquaredMinus > 0.0 {
            let inverse2A = 1.0 / (2.0 * A);
            let sqrt_term = sqrt(BsquaredMinus);
            let t1 = (-B + sqrt_term) * inverse2A;
            let t2 = (-B - sqrt_term) * inverse2A;
            let p1x = lvx * t1 + x0;
            let p1y = lvy * t1 + y0;
            let p1z = lvz * t1 + z0;
            if meets_all_bounds(p1x, p1y, p1z, bounds)
                && meets_all_bounds(p1x, p1y, p1z, more_bounds)
            {
                return true;
            }
            let p2x = lvx * t2 + x0;
            let p2y = lvy * t2 + y0;
            let p2z = lvz * t2 + z0;
            meets_all_bounds(p2x, p2y, p2z, bounds) && meets_all_bounds(p2x, p2y, p2z, more_bounds)
        } else {
            false
        }
    }

    /// `isFunctionallyIdentical(p)`.
    pub fn is_functionally_identical(&self, p: &Plane) -> bool {
        let cross1 = self.y * p.z - self.z * p.y;
        let cross2 = self.z * p.x - self.x * p.z;
        let cross3 = self.x * p.y - self.y * p.x;
        if cross1 * cross1 + cross2 * cross2 + cross3 * cross3 >= 5.0 * MINIMUM_RESOLUTION {
            return false;
        }
        let denom = 1.0 / (p.x * p.x + p.y * p.y + p.z * p.z);
        self.evaluate_is_zero_xyz(-p.x * p.D * denom, -p.y * p.D * denom, -p.z * p.D * denom)
    }

    /// `isNumericallyIdentical(Plane p)`.
    pub fn is_numerically_identical_plane(&self, p: &Plane) -> bool {
        let cross1 = self.y * p.z - self.z * p.y;
        let cross2 = self.z * p.x - self.x * p.z;
        let cross3 = self.x * p.y - self.y * p.x;
        if cross1 * cross1 + cross2 * cross2 + cross3 * cross3 >= MINIMUM_RESOLUTION_SQUARED {
            return false;
        }
        let denom = 1.0 / (p.x * p.x + p.y * p.y + p.z * p.z);
        self.evaluate_is_zero_xyz(-p.x * p.D * denom, -p.y * p.D * denom, -p.z * p.D * denom)
    }

    /// `findArcDistancePoints(planetModel, arcDistanceValue, startPoint,
    /// bounds)`: the points on the plane at the arc distance either way from
    /// `startPoint`.
    pub fn find_arc_distance_points(
        &self,
        pm: &PlanetModel,
        arc_distance_value: f64,
        start_point: &GeoPoint,
        bounds: &[&dyn Membership],
    ) -> Result<Vec<GeoPoint>> {
        if abs(self.D) >= MINIMUM_RESOLUTION {
            return Err(Error::IllegalState(
                "Can't find arc distance using plane that doesn't go through origin".into(),
            ));
        }
        if !self.evaluate_is_zero(start_point) {
            return Err(Error::IllegalArgument("Start point is not on plane".into()));
        }
        let azimuth_magnitude = sqrt(self.x * self.x + self.y * self.y);
        let cos_plane_altitude = self.z;
        let sin_plane_altitude = azimuth_magnitude;
        let cos_plane_azimuth = self.x / azimuth_magnitude;
        let sin_plane_azimuth = self.y / azimuth_magnitude;
        let x0 = start_point.x;
        let y0 = start_point.y;
        let z0 = start_point.z;
        let x1 = x0 * cos_plane_azimuth + y0 * sin_plane_azimuth;
        let y1 = -x0 * sin_plane_azimuth + y0 * cos_plane_azimuth;
        let z1 = z0;
        let x2 = x1 * cos_plane_altitude - z1 * sin_plane_altitude;
        let y2 = y1;
        let start_angle = atan2(y2, x2);
        let point1_angle = start_angle + arc_distance_value;
        let point2_angle = start_angle - arc_distance_value;
        let point1x2 = cos(point1_angle);
        let point1y2 = sin(point1_angle);
        let point1z2 = 0.0;
        let point2x2 = cos(point2_angle);
        let point2y2 = sin(point2_angle);
        let point2z2 = 0.0;
        let point1x1 = point1x2 * cos_plane_altitude + point1z2 * sin_plane_altitude;
        let point1y1 = point1y2;
        let point1z1 = -point1x2 * sin_plane_altitude + point1z2 * cos_plane_altitude;
        let point2x1 = point2x2 * cos_plane_altitude + point2z2 * sin_plane_altitude;
        let point2y1 = point2y2;
        let point2z1 = -point2x2 * sin_plane_altitude + point2z2 * cos_plane_altitude;
        let point1x0 = point1x1 * cos_plane_azimuth - point1y1 * sin_plane_azimuth;
        let point1y0 = point1x1 * sin_plane_azimuth + point1y1 * cos_plane_azimuth;
        let point1z0 = point1z1;
        let point2x0 = point2x1 * cos_plane_azimuth - point2y1 * sin_plane_azimuth;
        let point2y0 = point2x1 * sin_plane_azimuth + point2y1 * cos_plane_azimuth;
        let point2z0 = point2z1;
        let point1 = pm.create_surface_point_xyz(point1x0, point1y0, point1z0);
        let point2 = pm.create_surface_point_xyz(point2x0, point2y0, point2z0);
        let inside1 = meets_all_bounds(point1.x, point1.y, point1.z, bounds);
        let inside2 = meets_all_bounds(point2.x, point2.y, point2.z, bounds);
        Ok(match (inside1, inside2) {
            (true, true) => vec![point1, point2],
            (true, false) => vec![point1],
            (false, true) => vec![point2],
            (false, false) => Vec::new(),
        })
    }

    /// `getSampleIntersectionPoint(planetModel, q)`.
    pub fn sample_intersection_point(&self, pm: &PlanetModel, q: &Plane) -> Option<GeoPoint> {
        self.find_intersections_more(pm, q, NO_BOUNDS, NO_BOUNDS)
            .into_iter()
            .next()
    }

    /// `hashCode()`.
    pub fn java_hash_code(&self) -> i32 {
        let t = super::jmath::double_to_long_bits(self.D);
        self.v
            .java_hash_code()
            .wrapping_mul(31)
            .wrapping_add((t ^ (t >> 32)) as i32)
    }
}

/// The two-root tail of `findIntersections`/`findCrossings`.
#[allow(clippy::too_many_arguments)]
#[inline]
fn two_points(
    lvx: f64,
    lvy: f64,
    lvz: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    t1: f64,
    t2: f64,
    bounds: &[&dyn Membership],
    more_bounds: &[&dyn Membership],
) -> [Option<GeoPoint>; 2] {
    let p1x = lvx * t1 + x0;
    let p1y = lvy * t1 + y0;
    let p1z = lvz * t1 + z0;
    let p2x = lvx * t2 + x0;
    let p2y = lvy * t2 + y0;
    let p2z = lvz * t2 + z0;
    let v1 =
        meets_all_bounds(p1x, p1y, p1z, bounds) && meets_all_bounds(p1x, p1y, p1z, more_bounds);
    let v2 =
        meets_all_bounds(p2x, p2y, p2z, bounds) && meets_all_bounds(p2x, p2y, p2z, more_bounds);
    let p1 = v1.then(|| GeoPoint::new(p1x, p1y, p1z));
    let p2 = v2.then(|| GeoPoint::new(p2x, p2y, p2z));
    // In Java's order: the first valid point first.
    if p1.is_none() {
        [p2, None]
    } else {
        [p1, p2]
    }
}

/// `recordLineBounds(...)`.
#[allow(clippy::too_many_arguments)]
fn record_line_bounds(
    pm: &PlanetModel,
    bounds_info: &mut dyn Bounds,
    lvx: f64,
    lvy: f64,
    lvz: f64,
    x0: f64,
    y0: f64,
    z0: f64,
    bounds: &[&dyn Membership],
) {
    let (A, B, C) = line_quadratic(pm, lvx, lvy, lvz, x0, y0, z0);
    let BsquaredMinus = B * B - 4.0 * A * C;
    if abs(BsquaredMinus) < MINIMUM_RESOLUTION_SQUARED {
        let inverse2A = 1.0 / (2.0 * A);
        let t = -B * inverse2A;
        let px = lvx * t + x0;
        let py = lvy * t + y0;
        let pz = lvz * t + z0;
        if !meets_all_bounds(px, py, pz, bounds) {
            return;
        }
        bounds_info.add_point(&GeoPoint::new(px, py, pz));
    } else if BsquaredMinus > 0.0 {
        let inverse2A = 1.0 / (2.0 * A);
        let sqrt_term = sqrt(BsquaredMinus);
        let t1 = (-B + sqrt_term) * inverse2A;
        let t2 = (-B - sqrt_term) * inverse2A;
        let p1x = lvx * t1 + x0;
        let p1y = lvy * t1 + y0;
        let p1z = lvz * t1 + z0;
        let p2x = lvx * t2 + x0;
        let p2y = lvy * t2 + y0;
        let p2z = lvz * t2 + z0;
        let v1 = meets_all_bounds(p1x, p1y, p1z, bounds);
        let v2 = meets_all_bounds(p2x, p2y, p2z, bounds);
        if v1 {
            bounds_info.add_point(&GeoPoint::new(p1x, p1y, p1z));
        }
        if v2 {
            bounds_info.add_point(&GeoPoint::new(p2x, p2y, p2z));
        }
    } else {
        // If we can't intersect line with world, then it's outside the
        // world, so we have to assume everything is included.
        bounds_info.no_bound(pm);
    }
}

/// `addPoint(boundsInfo, bounds, point)`: only if inside every bound.
fn add_point(bounds_info: &mut dyn Bounds, bounds: &[&dyn Membership], point: &GeoPoint) {
    if !bounds.iter().all(|b| b.is_within(point)) {
        return;
    }
    bounds_info.add_point(point);
}

/// `modify(start, ...)`.
#[allow(clippy::too_many_arguments)]
fn modify(
    start: &GeoPoint,
    trans_x: f64,
    trans_y: f64,
    trans_z: f64,
    sin_ra: f64,
    cos_ra: f64,
    sin_ha: f64,
    cos_ha: f64,
) -> Vector {
    start
        .translate(trans_x, trans_y, trans_z)
        .rotate_xy_sc(sin_ra, cos_ra)
        .rotate_xz_sc(sin_ha, cos_ha)
}

/// `reverseModify(planetModel, point, ...)`.
#[allow(clippy::too_many_arguments)]
fn reverse_modify(
    pm: &PlanetModel,
    point: &Vector,
    trans_x: f64,
    trans_y: f64,
    trans_z: f64,
    sin_ra: f64,
    cos_ra: f64,
    sin_ha: f64,
    cos_ha: f64,
) -> GeoPoint {
    let result = point
        .rotate_xz_sc(-sin_ha, cos_ha)
        .rotate_xy_sc(-sin_ra, cos_ra)
        .translate(-trans_x, -trans_y, -trans_z);
    pm.create_surface_point_xyz(result.x, result.y, result.z)
}
