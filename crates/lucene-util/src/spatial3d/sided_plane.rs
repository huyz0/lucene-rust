//! `SidedPlane` (`org.apache.lucene.spatial3d.geom.SidedPlane`): a plane
//! that knows which side is "inside" -- the side a check point lies on --
//! and is a [`Membership`] for it.

#![allow(non_snake_case)]

use super::jmath::{abs, signum};
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;
use super::vector::{Vector, MINIMUM_RESOLUTION};
use super::{Error, Result};

/// A [`Plane`] (deref) with a side.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SidedPlane {
    /// The plane.
    pub plane: Plane,
    /// `sigNum`: the sign of `evaluate` on the inside.
    pub sig_num: f64,
}

impl std::ops::Deref for SidedPlane {
    type Target = Plane;

    #[inline]
    fn deref(&self) -> &Plane {
        &self.plane
    }
}

fn on_plane() -> Error {
    Error::IllegalArgument("Cannot determine sidedness because check point is on plane.".into())
}

/// `sigNum`, failing as Java's constructors do when it is zero (a NaN passes).
fn sided(plane: Plane, sig_num: f64) -> Result<SidedPlane> {
    if sig_num == 0.0 {
        return Err(on_plane());
    }
    Ok(SidedPlane { plane, sig_num })
}

impl SidedPlane {
    /// `SidedPlane(SidedPlane)`: the same plane, the other side.
    pub fn opposite(sided_plane: &SidedPlane) -> SidedPlane {
        SidedPlane {
            plane: Plane::with_d(&sided_plane.v, sided_plane.D),
            sig_num: -sided_plane.sig_num,
        }
    }

    /// `SidedPlane(pX, pY, pZ, A, B)`.
    pub fn from_xyz_vectors(
        pX: f64,
        pY: f64,
        pZ: f64,
        A: &Vector,
        B: &Vector,
    ) -> Result<SidedPlane> {
        let plane = Plane::from_vectors(A, B)?;
        sided(plane, signum(plane.evaluate_xyz(pX, pY, pZ)))
    }

    /// `SidedPlane(p, A, B)`: through the origin, `A` and `B`, with `p`
    /// inside.
    pub fn from_vectors(p: &Vector, A: &Vector, B: &Vector) -> Result<SidedPlane> {
        let plane = Plane::from_vectors(A, B)?;
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(A, B)`: through the origin, `A` and `B`, `sigNum` 1.
    pub fn from_two_vectors(A: &Vector, B: &Vector) -> Result<SidedPlane> {
        Ok(SidedPlane {
            plane: Plane::from_vectors(A, B)?,
            sig_num: 1.0,
        })
    }

    /// `SidedPlane(p, A, BX, BY, BZ)`.
    pub fn from_vector_xyz(
        p: &Vector,
        A: &Vector,
        BX: f64,
        BY: f64,
        BZ: f64,
    ) -> Result<SidedPlane> {
        let plane = Plane::from_vector_xyz(A, BX, BY, BZ)?;
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(p, onSide, A, B)`: `p` inside when `onSide`, outside
    /// otherwise.
    pub fn from_vectors_on_side(
        p: &Vector,
        on_side: bool,
        A: &Vector,
        B: &Vector,
    ) -> Result<SidedPlane> {
        let plane = Plane::from_vectors(A, B)?;
        let s = signum(plane.evaluate(p));
        sided(plane, if on_side { s } else { -s })
    }

    /// `SidedPlane(p, planetModel, sinLat)`: horizontal.
    pub fn horizontal(p: &Vector, pm: &PlanetModel, sin_lat: f64) -> Result<SidedPlane> {
        let plane = Plane::horizontal(pm, sin_lat);
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(p, x, y)`: vertical through the z axis and `(x, y)`.
    pub fn vertical(p: &Vector, x: f64, y: f64) -> Result<SidedPlane> {
        let plane = Plane::vertical(x, y);
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(p, vX, vY, vZ, D)`.
    pub fn from_abcd(p: &Vector, vX: f64, vY: f64, vZ: f64, D: f64) -> Result<SidedPlane> {
        let plane = Plane::new(vX, vY, vZ, D);
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(p, v, D)`.
    pub fn from_normal(p: &Vector, v: &Vector, D: f64) -> Result<SidedPlane> {
        let plane = Plane::with_d(v, D);
        sided(plane, signum(plane.evaluate(p)))
    }

    /// `SidedPlane(pX, pY, pZ, v, D)`.
    pub fn from_xyz_normal(pX: f64, pY: f64, pZ: f64, v: &Vector, D: f64) -> Result<SidedPlane> {
        let plane = Plane::with_d(v, D);
        sided(plane, signum(plane.evaluate_xyz(pX, pY, pZ)))
    }

    /// `constructNormalizedPerpendicularSidedPlane(insidePoint, normalVector,
    /// point1, point2)`: `None` where Java catches the failure (and returns
    /// null). Fails where Java's uncaught `Vector` constructor throws.
    pub fn construct_normalized_perpendicular_sided_plane(
        inside_point: &Vector,
        normal_vector: &Vector,
        point1: &Vector,
        point2: &Vector,
    ) -> Result<Option<SidedPlane>> {
        let points_vector = Vector::new(
            point1.x - point2.x,
            point1.y - point2.y,
            point1.z - point2.z,
        );
        let new_normal_vector = Vector::perpendicular(normal_vector, &points_vector)?;
        Ok(SidedPlane::from_normal(
            inside_point,
            &new_normal_vector,
            -new_normal_vector.dot_product(point1),
        )
        .ok())
    }

    /// `constructSidedPlaneFromTwoPoints(insidePoint, upperPoint,
    /// lowerPoint)`.
    pub fn construct_sided_plane_from_two_points(
        inside_point: &Vector,
        upper_point: &Vector,
        lower_point: &Vector,
    ) -> Result<SidedPlane> {
        let plane =
            Plane::construct_perpendicular_center_plane_two_points(upper_point, lower_point)?;
        SidedPlane::from_abcd(inside_point, plane.x, plane.y, plane.z, plane.D)
    }

    /// `constructSidedPlaneFromOnePoint(insidePoint, plane,
    /// intersectionPoint)`.
    pub fn construct_sided_plane_from_one_point(
        inside_point: &Vector,
        plane: &Plane,
        intersection_point: &Vector,
    ) -> Result<SidedPlane> {
        let new_plane =
            Plane::construct_perpendicular_center_plane_one_point(plane, intersection_point)?;
        SidedPlane::from_abcd(
            inside_point,
            new_plane.x,
            new_plane.y,
            new_plane.z,
            new_plane.D,
        )
    }

    /// `constructNormalizedThreePointSidedPlane(insidePoint, point1, point2,
    /// point3)`: tries the three ways of pairing the points' differences;
    /// `None` when all fail.
    pub fn construct_normalized_three_point_sided_plane(
        inside_point: &Vector,
        point1: &Vector,
        point2: &Vector,
        point3: &Vector,
    ) -> Option<SidedPlane> {
        let attempt = |a: [f64; 6], on: &Vector| -> Option<SidedPlane> {
            let normal = Vector::perpendicular_xyz(a[0], a[1], a[2], a[3], a[4], a[5]).ok()?;
            SidedPlane::from_normal(inside_point, &normal, -normal.dot_product(on)).ok()
        };
        attempt(
            [
                point1.x - point2.x,
                point1.y - point2.y,
                point1.z - point2.z,
                point2.x - point3.x,
                point2.y - point3.y,
                point2.z - point3.z,
            ],
            point2,
        )
        .or_else(|| {
            attempt(
                [
                    point1.x - point3.x,
                    point1.y - point3.y,
                    point1.z - point3.z,
                    point3.x - point2.x,
                    point3.y - point2.y,
                    point3.z - point2.z,
                ],
                point3,
            )
        })
        .or_else(|| {
            attempt(
                [
                    point3.x - point1.x,
                    point3.y - point1.y,
                    point3.z - point1.z,
                    point1.x - point2.x,
                    point1.y - point2.y,
                    point1.z - point2.z,
                ],
                point1,
            )
        })
    }

    /// `strictlyWithin(v)`: on the plane counts only when exactly on it.
    pub fn strictly_within(&self, v: &Vector) -> bool {
        self.strictly_within_xyz(v.x, v.y, v.z)
    }

    /// `strictlyWithin(x, y, z)`.
    pub fn strictly_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        let s = signum(self.evaluate_xyz(x, y, z));
        s == 0.0 || s == self.sig_num
    }

    /// `hashCode()`.
    pub fn java_hash_code(&self) -> i32 {
        let t = super::jmath::double_to_long_bits(self.sig_num);
        self.plane
            .java_hash_code()
            .wrapping_mul(31)
            .wrapping_add((t ^ (t >> 32)) as i32)
    }
}

impl Membership for SidedPlane {
    #[inline]
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        let eval_result = self.evaluate_xyz(x, y, z);
        if abs(eval_result) < MINIMUM_RESOLUTION {
            return true;
        }
        signum(eval_result) == self.sig_num
    }
}

impl std::fmt::Display for SidedPlane {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::geo::java_double_string as d;
        write!(
            f,
            "[A={}, B={}, C={}, D={}, side={}]",
            d(self.x),
            d(self.y),
            d(self.z),
            d(self.D),
            d(self.sig_num)
        )
    }
}
