//! geo3d's shape interfaces (`org.apache.lucene.spatial3d.geom`'s
//! `PlanetObject`, `SerializableObject`, `GeoBounds`, `GeoShape`,
//! `GeoOutsideDistance`, `GeoMembershipShape`, `GeoArea`, `GeoAreaShape`,
//! `GeoSizeable`, `GeoBBox`, `GeoDistance`, `GeoDistanceShape`, `GeoCircle`,
//! `GeoPath`, `GeoPolygon`, `GeoPointShape`) as traits with the same
//! supertrait lattice, and the shared bodies of their abstract base classes
//! (`GeoBaseBounds`, `GeoBaseMembershipShape`, `GeoBaseAreaShape`,
//! `GeoBaseDistanceShape`) as free functions each shape calls.
//!
//! Shapes are shared as `Arc<dyn Trait>`; a narrower trait object coerces to
//! a wider one (trait upcasting), as a Java reference widens.

use std::borrow::Cow;
use std::sync::Arc;

use super::bounds::{Bounded, Bounds};
use super::distance_style::DistanceStyle;
use super::geo_point::GeoPoint;
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;
use super::{Error, Result};

/// `SerializableObject`: geo3d's own stream format (see
/// [`super::serializable`]).
pub trait SerializableObject {
    /// `write(OutputStream)`: the object's fields, without its class.
    /// `BasePlanetObject`'s default throws `UnsupportedOperationException`.
    fn write(&self, out: &mut Vec<u8>) -> Result<()>;

    /// The `StandardObjects` registry code `writeClass` writes for this
    /// class, or `None` for an unregistered class.
    fn class_code(&self) -> Option<u8>;
}

/// `PlanetObject`: an object on a planet.
pub trait PlanetObject: SerializableObject {
    /// `getPlanetModel()`.
    fn planet_model(&self) -> &Arc<PlanetModel>;

    /// The concrete object, for Java's `instanceof` on a shape a factory
    /// returned behind an interface.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// `GeoBounds`.
pub trait GeoBounds: Bounded + Membership + PlanetObject {}

/// `GeoShape`: a shape with edge points that can be tested against a plane.
pub trait GeoShape: GeoBounds {
    /// `getEdgePoints()`: at least one point on each disconnected edge.
    fn edge_points(&self) -> Cow<'_, [GeoPoint]>;

    /// `intersects(plane, notablePoints, bounds)`: whether the shape's edge
    /// meets the bounded plane.
    fn intersects(
        &self,
        plane: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool;
}

/// `GeoOutsideDistance`: distances to points outside the shape.
pub trait GeoOutsideDistance: Membership {
    /// `computeOutsideDistance(distanceStyle, x, y, z)`: 0 inside.
    fn compute_outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64;

    /// `computeOutsideDistance(distanceStyle, point)`.
    fn compute_outside_distance_to(&self, style: DistanceStyle, point: &GeoPoint) -> f64 {
        self.compute_outside_distance(style, point.x, point.y, point.z)
    }
}

/// `GeoMembershipShape`.
pub trait GeoMembershipShape: GeoShape + GeoOutsideDistance {}

/// `GeoArea`'s relationship constants, as an enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum GeoAreaRelationship {
    /// `GeoArea.CONTAINS`: the area contains the shape.
    Contains = 0,
    /// `GeoArea.WITHIN`: the area is within the shape.
    Within = 1,
    /// `GeoArea.OVERLAPS`.
    Overlaps = 2,
    /// `GeoArea.DISJOINT`.
    Disjoint = 3,
}

/// `GeoArea`: an area that can be related to a shape.
pub trait GeoArea: Membership {
    /// `getRelationship(shape)`. Fails (as Java throws) when the planet
    /// models differ.
    fn get_relationship(&self, shape: &dyn GeoShape) -> Result<GeoAreaRelationship>;
}

/// A `GeoArea` that is also a `PlanetObject` -- what `GeoAreaFactory`
/// returns (every Java implementation is both). Implemented for every type
/// that is both.
pub trait GeoAreaObject: GeoArea + PlanetObject {}

impl<T: GeoArea + PlanetObject> GeoAreaObject for T {}

/// `GeoAreaShape`.
pub trait GeoAreaShape: GeoMembershipShape + GeoAreaObject {
    /// `intersects(GeoShape)`: whether the two shapes' edges cross.
    fn intersects_shape(&self, shape: &dyn GeoShape) -> bool;
}

/// `GeoSizeable`.
pub trait GeoSizeable {
    /// `getRadius()`.
    fn radius(&self) -> f64;
    /// `getCenter()`.
    fn center(&self) -> GeoPoint;
}

/// `GeoBBox`.
pub trait GeoBBox: GeoAreaShape + GeoSizeable {
    /// `expand(angle)`: a box grown by `angle` on every side.
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>>;
}

/// `GeoDistance`: distances to points inside the shape.
pub trait GeoDistance: Membership {
    /// `computeDistance(distanceStyle, x, y, z)`: infinity outside.
    fn compute_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64;

    /// `computeDistance(distanceStyle, point)`.
    fn compute_distance_to(&self, style: DistanceStyle, point: &GeoPoint) -> f64 {
        self.compute_distance(style, point.x, point.y, point.z)
    }

    /// `computeDeltaDistance(distanceStyle, x, y, z)`: twice the distance by
    /// default.
    fn compute_delta_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        self.compute_distance(style, x, y, z) * 2.0
    }
}

/// `GeoDistanceShape`.
pub trait GeoDistanceShape: GeoAreaShape + GeoDistance {
    /// `getDistanceBounds(bounds, distanceStyle, distanceValue)`: the bounds
    /// of the part of the shape within the distance. Fails for a distance
    /// style without a reverse mapping, as Java throws.
    fn get_distance_bounds(
        &self,
        bounds: &mut dyn Bounds,
        style: DistanceStyle,
        distance_value: f64,
    ) -> Result<()>;
}

/// `GeoCircle`.
pub trait GeoCircle: GeoDistanceShape + GeoSizeable {}

/// `GeoPath`.
pub trait GeoPath: GeoDistanceShape {
    /// `computeNearestDistance(distanceStyle, x, y, z)`: the distance along
    /// the path to the point nearest `(x, y, z)`.
    fn compute_nearest_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64;

    /// `computePathCenterDistance(distanceStyle, x, y, z)`.
    fn compute_path_center_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64;
}

/// `GeoPolygon`.
pub trait GeoPolygon: GeoAreaShape {}

/// `GeoPointShape`.
pub trait GeoPointShape: GeoCircle + GeoBBox {}

/// `GeoBaseBounds.getBounds`: the poles and axis points the shape contains.
pub(crate) fn base_get_bounds<S: Membership + ?Sized>(
    shape: &S,
    pm: &PlanetModel,
    bounds: &mut dyn Bounds,
) {
    if shape.is_within(&pm.north_pole) {
        bounds
            .no_top_latitude_bound()
            .no_longitude_bound()
            .add_point(&pm.north_pole);
    }
    if shape.is_within(&pm.south_pole) {
        bounds
            .no_bottom_latitude_bound()
            .no_longitude_bound()
            .add_point(&pm.south_pole);
    }
    if shape.is_within(&pm.min_x_pole) {
        bounds.add_point(&pm.min_x_pole);
    }
    if shape.is_within(&pm.max_x_pole) {
        bounds.add_point(&pm.max_x_pole);
    }
    if shape.is_within(&pm.min_y_pole) {
        bounds.add_point(&pm.min_y_pole);
    }
    if shape.is_within(&pm.max_y_pole) {
        bounds.add_point(&pm.max_y_pole);
    }
}

/// `GeoBaseAreaShape`'s `ALL_INSIDE`/`SOME_INSIDE`/`NONE_INSIDE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Inside {
    All,
    Some,
    None,
}

/// `isShapeInsideGeoAreaShape(geoShape)`: how many of `shape`'s edge points
/// are inside `area`.
pub(crate) fn is_shape_inside_area<A: Membership + ?Sized>(
    area: &A,
    shape: &dyn GeoShape,
) -> Inside {
    edge_points_inside(&shape.edge_points(), area)
}

/// `isGeoAreaShapeInsideShape(geoshape)`: how many of `area_edge_points` are
/// inside `shape`.
pub(crate) fn is_area_inside_shape(area_edge_points: &[GeoPoint], shape: &dyn GeoShape) -> Inside {
    edge_points_inside(area_edge_points, shape)
}

fn edge_points_inside<M: Membership + ?Sized>(points: &[GeoPoint], m: &M) -> Inside {
    let mut found_outside = false;
    let mut found_inside = false;
    for p in points {
        if m.is_within(p) {
            found_inside = true;
        } else {
            found_outside = true;
        }
        if found_inside && found_outside {
            return Inside::Some;
        }
    }
    if !found_inside && !found_outside {
        return Inside::None;
    }
    if found_inside && !found_outside {
        return Inside::All;
    }
    if found_outside && !found_inside {
        return Inside::None;
    }
    Inside::Some
}

/// Java's "Cannot relate shapes with different planet models." check.
pub(crate) fn check_same_planet(a: &PlanetModel, shape: &dyn GeoShape) -> Result<()> {
    if **shape.planet_model() != *a {
        return Err(Error::IllegalArgument(
            "Cannot relate shapes with different planet models.".into(),
        ));
    }
    Ok(())
}

/// `GeoBaseAreaShape.getRelationship(geoShape)`.
pub(crate) fn base_get_relationship<A: GeoAreaShape + ?Sized>(
    area: &A,
    shape: &dyn GeoShape,
) -> Result<GeoAreaRelationship> {
    check_same_planet(area.planet_model(), shape)?;
    let inside_geo_area_shape = is_shape_inside_area(area, shape);
    if inside_geo_area_shape == Inside::Some {
        return Ok(GeoAreaRelationship::Overlaps);
    }
    let inside_shape = is_area_inside_shape(&area.edge_points(), shape);
    if inside_shape == Inside::Some {
        return Ok(GeoAreaRelationship::Overlaps);
    }
    if inside_geo_area_shape == Inside::All && inside_shape == Inside::All {
        return Ok(GeoAreaRelationship::Overlaps);
    }
    if area.intersects_shape(shape) {
        return Ok(GeoAreaRelationship::Overlaps);
    }
    if inside_geo_area_shape == Inside::All {
        return Ok(GeoAreaRelationship::Within);
    }
    if inside_shape == Inside::All {
        return Ok(GeoAreaRelationship::Contains);
    }
    Ok(GeoAreaRelationship::Disjoint)
}

/// `PlanetObject` + `GeoBounds` for a shape with a `planet_model` field.
macro_rules! impl_planet_object {
    ($t:ty) => {
        impl $crate::spatial3d::shape::PlanetObject for $t {
            fn planet_model(
                &self,
            ) -> &std::sync::Arc<$crate::spatial3d::planet_model::PlanetModel> {
                &self.planet_model
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
        impl $crate::spatial3d::shape::GeoBounds for $t {}
    };
}
pub(crate) use impl_planet_object;

/// `GeoBaseMembershipShape`: `computeOutsideDistance` is 0 inside, the
/// shape's own `outside_distance` otherwise.
macro_rules! impl_membership_shape {
    ($t:ty) => {
        impl $crate::spatial3d::shape::GeoMembershipShape for $t {}
        impl $crate::spatial3d::shape::GeoOutsideDistance for $t {
            fn compute_outside_distance(
                &self,
                style: $crate::spatial3d::distance_style::DistanceStyle,
                x: f64,
                y: f64,
                z: f64,
            ) -> f64 {
                if $crate::spatial3d::membership::Membership::is_within_xyz(self, x, y, z) {
                    0.0
                } else {
                    self.outside_distance(style, x, y, z)
                }
            }
        }
    };
}
pub(crate) use impl_membership_shape;

/// `GeoBaseAreaShape.getRelationship` for a shape that does not override it.
macro_rules! impl_base_area {
    ($t:ty) => {
        impl $crate::spatial3d::shape::GeoArea for $t {
            fn get_relationship(
                &self,
                shape: &dyn $crate::spatial3d::shape::GeoShape,
            ) -> $crate::spatial3d::Result<$crate::spatial3d::shape::GeoAreaRelationship> {
                $crate::spatial3d::shape::base_get_relationship(self, shape)
            }
        }
    };
}
pub(crate) use impl_base_area;

/// `GeoBaseDistanceShape`: `computeDistance`/`computeDeltaDistance` are
/// infinite outside, the shape's own `distance`/`delta_distance` inside;
/// `getDistanceBounds` with an infinite distance is the shape's bounds.
macro_rules! impl_distance_shape {
    ($t:ty) => {
        impl $crate::spatial3d::shape::GeoDistance for $t {
            fn compute_distance(
                &self,
                style: $crate::spatial3d::distance_style::DistanceStyle,
                x: f64,
                y: f64,
                z: f64,
            ) -> f64 {
                if !$crate::spatial3d::membership::Membership::is_within_xyz(self, x, y, z) {
                    return f64::INFINITY;
                }
                self.distance(style, x, y, z)
            }

            fn compute_delta_distance(
                &self,
                style: $crate::spatial3d::distance_style::DistanceStyle,
                x: f64,
                y: f64,
                z: f64,
            ) -> f64 {
                if !$crate::spatial3d::membership::Membership::is_within_xyz(self, x, y, z) {
                    return f64::INFINITY;
                }
                self.delta_distance(style, x, y, z)
            }
        }
        impl $crate::spatial3d::shape::GeoDistanceShape for $t {
            fn get_distance_bounds(
                &self,
                bounds: &mut dyn $crate::spatial3d::bounds::Bounds,
                style: $crate::spatial3d::distance_style::DistanceStyle,
                distance_value: f64,
            ) -> $crate::spatial3d::Result<()> {
                if distance_value == f64::INFINITY {
                    $crate::spatial3d::bounds::Bounded::get_bounds(self, bounds);
                    return Ok(());
                }
                self.distance_bounds(bounds, style, distance_value)
            }
        }
    };
}
pub(crate) use impl_distance_shape;
