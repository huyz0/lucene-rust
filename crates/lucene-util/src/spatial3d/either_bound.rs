//! The `EitherBound` inner class of the wide shapes (`GeoWideRectangle`,
//! `GeoWideNorthRectangle`, `GeoWideSouthRectangle`,
//! `GeoWideDegenerateHorizontalLine`, `GeoWideLongitudeSlice`): inside
//! either of two planes -- a longitude range wider than half the planet.

use super::membership::Membership;
use super::sided_plane::SidedPlane;

/// Inside the left plane or the right plane.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EitherBound {
    pub(crate) left: SidedPlane,
    pub(crate) right: SidedPlane,
}

impl Membership for EitherBound {
    #[inline]
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.left.is_within_xyz(x, y, z) || self.right.is_within_xyz(x, y, z)
    }
}
