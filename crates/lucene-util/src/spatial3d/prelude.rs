//! The imports every shape module shares.

pub(crate) use std::borrow::Cow;
pub(crate) use std::f64::consts::PI;
pub(crate) use std::sync::Arc;

pub(crate) use super::bounds::{Bounded, Bounds};
pub(crate) use super::distance_style::DistanceStyle;
pub(crate) use super::geo_point::GeoPoint;
pub(crate) use super::jmath::{abs, cos, max, min, sin, sqrt};
pub(crate) use super::membership::Membership;
pub(crate) use super::plane::Plane;
pub(crate) use super::planet_model::PlanetModel;
pub(crate) use super::serializable::{read_double, write_double, Input};
pub(crate) use super::shape::{
    base_get_bounds, impl_base_area, impl_membership_shape, impl_planet_object, GeoArea,
    GeoAreaRelationship, GeoAreaShape, GeoBBox, GeoShape, GeoSizeable, SerializableObject,
};
pub(crate) use super::sided_plane::SidedPlane;
pub(crate) use super::vector::{Vector, MINIMUM_ANGULAR_RESOLUTION, MINIMUM_RESOLUTION};
pub(crate) use super::{Error, Result};

/// `new IllegalArgumentException(msg)`.
pub(crate) fn illegal(msg: &str) -> Error {
    Error::IllegalArgument(msg.into())
}
