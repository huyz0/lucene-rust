//! `Bounds` and `Bounded` (`org.apache.lucene.spatial3d.geom`): the visitor a
//! shape reports its extent to, implemented by [`LatLonBounds`] and
//! [`XYZBounds`].
//!
//! Java's methods return `this` for chaining; here they return
//! `&mut dyn Bounds`.
//!
//! [`LatLonBounds`]: super::lat_lon_bounds::LatLonBounds
//! [`XYZBounds`]: super::xyz_bounds::XYZBounds

use super::geo_point::GeoPoint;
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;
use super::xyz_bounds::XYZBounds;

/// `Bounds`.
pub trait Bounds {
    /// `addPlane(planetModel, plane, bounds)`.
    fn add_plane(
        &mut self,
        pm: &PlanetModel,
        plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds;
    /// `addHorizontalPlane(planetModel, latitude, horizontalPlane, bounds)`.
    fn add_horizontal_plane(
        &mut self,
        pm: &PlanetModel,
        latitude: f64,
        horizontal_plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds;
    /// `addVerticalPlane(planetModel, longitude, verticalPlane, bounds)`.
    fn add_vertical_plane(
        &mut self,
        pm: &PlanetModel,
        longitude: f64,
        vertical_plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds;
    /// `addIntersection(planetModel, plane1, plane2, bounds)`.
    fn add_intersection(
        &mut self,
        pm: &PlanetModel,
        plane1: &Plane,
        plane2: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds;
    /// `addPoint(point)`.
    fn add_point(&mut self, point: &GeoPoint) -> &mut dyn Bounds;
    /// `addXValue(point)`.
    fn add_x_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds;
    /// `addYValue(point)`.
    fn add_y_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds;
    /// `addZValue(point)`.
    fn add_z_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds;
    /// `isWide()`.
    fn is_wide(&mut self) -> &mut dyn Bounds;
    /// `noLongitudeBound()`.
    fn no_longitude_bound(&mut self) -> &mut dyn Bounds;
    /// `noTopLatitudeBound()`.
    fn no_top_latitude_bound(&mut self) -> &mut dyn Bounds;
    /// `noBottomLatitudeBound()`.
    fn no_bottom_latitude_bound(&mut self) -> &mut dyn Bounds;
    /// `noBound(planetModel)`.
    fn no_bound(&mut self, pm: &PlanetModel) -> &mut dyn Bounds;
    /// `bounds instanceof XYZBounds`: the one runtime type test geo3d makes
    /// (`GeoStandardPath`).
    fn as_xyz_bounds(&mut self) -> Option<&mut XYZBounds> {
        None
    }
}

/// `Bounded`: reports its extent to a [`Bounds`].
pub trait Bounded {
    /// `getBounds(bounds)`.
    fn get_bounds(&self, bounds: &mut dyn Bounds);
}
