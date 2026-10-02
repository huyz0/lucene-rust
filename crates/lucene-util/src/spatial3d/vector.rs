//! `Vector` (`org.apache.lucene.spatial3d.geom.Vector`): a 3d vector from the
//! origin, and the resolution constants every geo3d comparison is made
//! against.

use super::jmath::{abs, sin_cos, sqrt};
use super::membership::Membership;
use super::planet_model::PlanetModel;
use super::{Error, Result};

/// Values whose magnitude is below this are taken to be zero.
pub const MINIMUM_RESOLUTION: f64 = 1.0e-12;
/// Angular version of [`MINIMUM_RESOLUTION`].
pub const MINIMUM_ANGULAR_RESOLUTION: f64 = std::f64::consts::PI * MINIMUM_RESOLUTION;
/// For squared quantities, the bound is squared too.
pub const MINIMUM_RESOLUTION_SQUARED: f64 = MINIMUM_RESOLUTION * MINIMUM_RESOLUTION;
/// For cubed quantities, cube the bound.
pub const MINIMUM_RESOLUTION_CUBED: f64 = MINIMUM_RESOLUTION_SQUARED * MINIMUM_RESOLUTION;
/// Gram-Schmidt convergence envelope.
const MINIMUM_GRAM_SCHMIDT_ENVELOPE: f64 = MINIMUM_RESOLUTION * 0.5;

/// A 3d vector in space, not necessarily going through the origin.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vector {
    /// The x value.
    pub x: f64,
    /// The y value.
    pub y: f64,
    /// The z value.
    pub z: f64,
}

/// `Vector(AX, AY, AZ, BX, BY, BZ)`'s Gram-Schmidt loop: a normalized vector
/// perpendicular to both inputs, or `None` where Java throws.
fn perpendicular(
    ax: f64,
    ay: f64,
    az: f64,
    bx: f64,
    by: f64,
    bz: f64,
) -> std::result::Result<(f64, f64, f64), &'static str> {
    // Compute the naive perpendicular
    let this_x = ay * bz - az * by;
    let this_y = az * bx - ax * bz;
    let this_z = ax * by - ay * bx;
    let magnitude = Vector::magnitude_of(this_x, this_y, this_z);
    if magnitude == 0.0 {
        return Err("Degenerate/parallel vector constructed");
    }
    let inverse_magnitude = 1.0 / magnitude;
    let mut normalize_x = this_x * inverse_magnitude;
    let mut normalize_y = this_y * inverse_magnitude;
    let mut normalize_z = this_z * inverse_magnitude;
    // For a plane to work, the dot product between the normal vector and the
    // points needs to be less than the minimum resolution.
    let mut i = 0;
    loop {
        let current_dot_prod_a = ax * normalize_x + ay * normalize_y + az * normalize_z;
        let current_dot_prod_b = bx * normalize_x + by * normalize_y + bz * normalize_z;
        if abs(current_dot_prod_a) < MINIMUM_GRAM_SCHMIDT_ENVELOPE
            && abs(current_dot_prod_b) < MINIMUM_GRAM_SCHMIDT_ENVELOPE
        {
            break;
        }
        // Converge on the one that has largest dot product
        let (cx, cy, cz, current_dot_prod) = if abs(current_dot_prod_a) > abs(current_dot_prod_b) {
            (ax, ay, az, current_dot_prod_a)
        } else {
            (bx, by, bz, current_dot_prod_b)
        };
        // Adjust
        normalize_x -= current_dot_prod * cx;
        normalize_y -= current_dot_prod * cy;
        normalize_z -= current_dot_prod * cz;
        // Normalize
        let corrected_magnitude = Vector::magnitude_of(normalize_x, normalize_y, normalize_z);
        let inverse_corrected_magnitude = 1.0 / corrected_magnitude;
        normalize_x *= inverse_corrected_magnitude;
        normalize_y *= inverse_corrected_magnitude;
        normalize_z *= inverse_corrected_magnitude;
        // `if (i++ > 10)`
        let over = i > 10;
        i += 1;
        if over {
            return Err("Plane could not be constructed! Could not find a normal vector.");
        }
    }
    Ok((normalize_x, normalize_y, normalize_z))
}

impl Vector {
    /// `Vector(x, y, z)`.
    #[inline]
    pub const fn new(x: f64, y: f64, z: f64) -> Vector {
        Vector { x, y, z }
    }

    /// `Vector(AX, AY, AZ, BX, BY, BZ)`: the normalized vector perpendicular
    /// to two (non-zero, non-parallel) vectors, refined by Gram-Schmidt.
    pub fn perpendicular_xyz(
        ax: f64,
        ay: f64,
        az: f64,
        bx: f64,
        by: f64,
        bz: f64,
    ) -> Result<Vector> {
        perpendicular(ax, ay, az, bx, by, bz)
            .map(|(x, y, z)| Vector { x, y, z })
            .map_err(|m| Error::IllegalArgument(m.into()))
    }

    /// `Vector(A, BX, BY, BZ)`.
    pub fn perpendicular_to(a: &Vector, bx: f64, by: f64, bz: f64) -> Result<Vector> {
        Self::perpendicular_xyz(a.x, a.y, a.z, bx, by, bz)
    }

    /// `Vector(A, B)`.
    pub fn perpendicular(a: &Vector, b: &Vector) -> Result<Vector> {
        Self::perpendicular_xyz(a.x, a.y, a.z, b.x, b.y, b.z)
    }

    /// `Vector.magnitude(x, y, z)`.
    #[inline]
    pub fn magnitude_of(x: f64, y: f64, z: f64) -> f64 {
        sqrt(x * x + y * y + z * z)
    }

    /// `normalize()`: the unit vector, or `None` for a (near-)zero vector.
    pub fn normalize(&self) -> Option<Vector> {
        let denom = self.magnitude();
        if denom < MINIMUM_RESOLUTION {
            return None;
        }
        let norm_factor = 1.0 / denom;
        Some(Vector::new(
            self.x * norm_factor,
            self.y * norm_factor,
            self.z * norm_factor,
        ))
    }

    /// `crossProductEvaluateIsZero(A, B, point)`: whether `point` lies on
    /// the plane through the origin, `A` and `B`.
    pub fn cross_product_evaluate_is_zero(a: &Vector, b: &Vector, point: &Vector) -> Result<bool> {
        let this_x = a.y * b.z - a.z * b.y;
        let this_y = a.z * b.x - a.x * b.z;
        let this_z = a.x * b.y - a.y * b.x;
        if Vector::magnitude_of(this_x, this_y, this_z) == 0.0 {
            return Ok(true);
        }
        let (nx, ny, nz) = perpendicular(a.x, a.y, a.z, b.x, b.y, b.z)
            .map_err(|m| Error::IllegalArgument(m.into()))?;
        Ok(abs(nx * point.x + ny * point.y + nz * point.z) < MINIMUM_RESOLUTION)
    }

    /// `dotProduct(v)`.
    #[inline]
    pub fn dot_product(&self, v: &Vector) -> f64 {
        self.x * v.x + self.y * v.y + self.z * v.z
    }

    /// `dotProduct(x, y, z)`.
    #[inline]
    pub fn dot_product_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        self.x * x + self.y * y + self.z * z
    }

    /// `isWithin(bounds, moreBounds)`: inside every bound.
    pub fn is_within_bounds(
        &self,
        bounds: &[&dyn Membership],
        more_bounds: &[&dyn Membership],
    ) -> bool {
        bounds.iter().all(|b| b.is_within(self)) && more_bounds.iter().all(|b| b.is_within(self))
    }

    /// `translate(xOffset, yOffset, zOffset)`.
    pub fn translate(&self, x_offset: f64, y_offset: f64, z_offset: f64) -> Vector {
        Vector::new(self.x - x_offset, self.y - y_offset, self.z - z_offset)
    }

    /// `rotateXY(angle)`.
    pub fn rotate_xy(&self, angle: f64) -> Vector {
        let (s, c) = sin_cos(angle);
        self.rotate_xy_sc(s, c)
    }

    /// `rotateXY(sinAngle, cosAngle)`.
    pub fn rotate_xy_sc(&self, sin_angle: f64, cos_angle: f64) -> Vector {
        Vector::new(
            self.x * cos_angle - self.y * sin_angle,
            self.x * sin_angle + self.y * cos_angle,
            self.z,
        )
    }

    /// `rotateXZ(angle)`.
    pub fn rotate_xz(&self, angle: f64) -> Vector {
        let (s, c) = sin_cos(angle);
        self.rotate_xz_sc(s, c)
    }

    /// `rotateXZ(sinAngle, cosAngle)`.
    pub fn rotate_xz_sc(&self, sin_angle: f64, cos_angle: f64) -> Vector {
        Vector::new(
            self.x * cos_angle - self.z * sin_angle,
            self.y,
            self.x * sin_angle + self.z * cos_angle,
        )
    }

    /// `rotateZY(angle)`.
    pub fn rotate_zy(&self, angle: f64) -> Vector {
        let (s, c) = sin_cos(angle);
        self.rotate_zy_sc(s, c)
    }

    /// `rotateZY(sinAngle, cosAngle)`.
    pub fn rotate_zy_sc(&self, sin_angle: f64, cos_angle: f64) -> Vector {
        Vector::new(
            self.x,
            self.z * sin_angle + self.y * cos_angle,
            self.z * cos_angle - self.y * sin_angle,
        )
    }

    /// `linearDistanceSquared(v)`.
    #[inline]
    pub fn linear_distance_squared(&self, v: &Vector) -> f64 {
        self.linear_distance_squared_xyz(v.x, v.y, v.z)
    }

    /// `linearDistanceSquared(x, y, z)`.
    #[inline]
    pub fn linear_distance_squared_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        let delta_x = self.x - x;
        let delta_y = self.y - y;
        let delta_z = self.z - z;
        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
    }

    /// `linearDistance(v)`.
    #[inline]
    pub fn linear_distance(&self, v: &Vector) -> f64 {
        sqrt(self.linear_distance_squared(v))
    }

    /// `linearDistance(x, y, z)`.
    #[inline]
    pub fn linear_distance_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        sqrt(self.linear_distance_squared_xyz(x, y, z))
    }

    /// `normalDistanceSquared(v)`.
    #[inline]
    pub fn normal_distance_squared(&self, v: &Vector) -> f64 {
        self.normal_distance_squared_xyz(v.x, v.y, v.z)
    }

    /// `normalDistanceSquared(x, y, z)`.
    #[inline]
    pub fn normal_distance_squared_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        let t = self.dot_product_xyz(x, y, z);
        let delta_x = self.x * t - x;
        let delta_y = self.y * t - y;
        let delta_z = self.z * t - z;
        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z
    }

    /// `normalDistance(v)`.
    #[inline]
    pub fn normal_distance(&self, v: &Vector) -> f64 {
        sqrt(self.normal_distance_squared(v))
    }

    /// `normalDistance(x, y, z)`.
    #[inline]
    pub fn normal_distance_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        sqrt(self.normal_distance_squared_xyz(x, y, z))
    }

    /// `magnitude()`.
    #[inline]
    pub fn magnitude(&self) -> f64 {
        Vector::magnitude_of(self.x, self.y, self.z)
    }

    /// `isNumericallyIdentical(x, y, z)`.
    #[inline]
    pub fn is_numerically_identical_xyz(&self, other_x: f64, other_y: f64, other_z: f64) -> bool {
        let delta_x = self.x - other_x;
        let delta_y = self.y - other_y;
        let delta_z = self.z - other_z;
        delta_x * delta_x + delta_y * delta_y + delta_z * delta_z < MINIMUM_RESOLUTION_SQUARED
    }

    /// `isNumericallyIdentical(other)`.
    #[inline]
    pub fn is_numerically_identical(&self, other: &Vector) -> bool {
        self.is_numerically_identical_xyz(other.x, other.y, other.z)
    }

    /// `isParallel(x, y, z)`.
    pub fn is_parallel_xyz(&self, other_x: f64, other_y: f64, other_z: f64) -> bool {
        let this_x = self.y * other_z - self.z * other_y;
        let this_y = self.z * other_x - self.x * other_z;
        let this_z = self.x * other_y - self.y * other_x;
        this_x * this_x + this_y * this_y + this_z * this_z < MINIMUM_RESOLUTION_SQUARED
    }

    /// `isParallel(other)`.
    pub fn is_parallel(&self, other: &Vector) -> bool {
        self.is_parallel_xyz(other.x, other.y, other.z)
    }

    /// `computeDesiredEllipsoidMagnitude(planetModel, x, y, z)`.
    #[inline]
    pub fn compute_desired_ellipsoid_magnitude(pm: &PlanetModel, x: f64, y: f64, z: f64) -> f64 {
        1.0 / sqrt(
            x * x * pm.inverse_xy_scaling_squared
                + y * y * pm.inverse_xy_scaling_squared
                + z * z * pm.inverse_z_scaling_squared,
        )
    }

    /// `computeDesiredEllipsoidMagnitude(planetModel, z)`.
    #[inline]
    pub fn compute_desired_ellipsoid_magnitude_z(pm: &PlanetModel, z: f64) -> f64 {
        1.0 / sqrt(
            (1.0 - z * z) * pm.inverse_xy_scaling_squared + z * z * pm.inverse_z_scaling_squared,
        )
    }

    /// `hashCode()`.
    pub fn java_hash_code(&self) -> i32 {
        let h = |v: f64| {
            let t = super::jmath::double_to_long_bits(v);
            (t ^ (t >> 32)) as i32
        };
        let mut result = h(self.x);
        result = result.wrapping_mul(31).wrapping_add(h(self.y));
        result.wrapping_mul(31).wrapping_add(h(self.z))
    }
}

impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use crate::geo::java_double_string as d;
        write!(f, "[X={}, Y={}, Z={}]", d(self.x), d(self.y), d(self.z))
    }
}
