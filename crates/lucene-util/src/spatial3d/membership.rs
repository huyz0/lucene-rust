//! `Membership` (`org.apache.lucene.spatial3d.geom.Membership`): anything
//! that can say whether a point is inside it.

use super::vector::Vector;

/// Implemented by shapes, sided planes and solids.
pub trait Membership: Send + Sync {
    /// `isWithin(x, y, z)`.
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool;

    /// `isWithin(Vector)`.
    #[inline]
    fn is_within(&self, point: &Vector) -> bool {
        self.is_within_xyz(point.x, point.y, point.z)
    }
}

/// `true` when `(x, y, z)` is inside every bound (`Plane.meetsAllBounds`).
#[inline]
pub(crate) fn meets_all_bounds(x: f64, y: f64, z: f64, bounds: &[&dyn Membership]) -> bool {
    meets_all_bounds_of(x, y, z, bounds)
}

/// [`meets_all_bounds`] over bounds of one concrete type: a path segment's
/// or a polygon edge's bounds are all `SidedPlane`s, and calling them
/// statically lets the membership test inline into the distance loops,
/// where HotSpot inlines Java's monomorphic `isWithin` call site.
#[inline]
pub(crate) fn meets_all_bounds_of<M: Membership + ?Sized>(
    x: f64,
    y: f64,
    z: f64,
    bounds: &[&M],
) -> bool {
    bounds.iter().all(|b| b.is_within_xyz(x, y, z))
}
