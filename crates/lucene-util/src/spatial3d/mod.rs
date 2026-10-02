//! Port of Lucene 10.5.0's `lucene-spatial3d` geometry
//! (`org.apache.lucene.spatial3d.geom`): shapes on an ellipsoid in 3d
//! Cartesian space -- planet models, planes and sided planes, latitude and
//! longitude zones and slices, bounding boxes, circles, paths, convex,
//! concave, complex and composite polygons, x/y/z solids -- their
//! relationships, bounds and distances, and the shape serialization.
//!
//! Like `geo`, it is pure geometry with no index dependency, so it lives in
//! `lucene-util`, the lowest crate `Geo3DPoint`'s field (`lucene-index`) and
//! query (`lucene-search`) can reach.
//!
//! # Fidelity
//!
//! Every relation test is a comparison against [`vector::MINIMUM_RESOLUTION`]
//! of an expression computed in floating point, so every function keeps
//! Java's expressions as written -- operand order, association, `Math.min`'s
//! NaN rule -- and Java's variable names where they are the math's (`A0`,
//! `B1`, `D`; hence `non_snake_case` is allowed in the modules that use
//! them). `crates/lucene-util/tests/geo3d_fixtures.rs` compares results bit
//! for bit against real Lucene over a seeded random corpus.
//!
//! **Trigonometry.** Java's `Math.sin`/`cos`/`tan` are HotSpot intrinsics on
//! x86-64 whose last bit differs from `StrictMath`'s fdlibm for ~3.4% of
//! arguments; this port uses fdlibm ([`jmath`]), which is what `Math.sin`
//! returns with the intrinsics disabled -- and how the fixtures are made
//! (`scripts/gen-fixtures.sh` runs `GenGeo3d` with
//! `-XX:DisableIntrinsic=_dsin,_dcos,_dtan`). Against a stock x86-64 JVM a
//! point's coordinates can therefore differ by an ulp, which can flip a
//! relation only for a point within an ulp of a shape's edge. Measured on
//! JDK 25/x86-64 by running `GenGeo3d` once more without the flags: 5 824
//! of its 50 571 records differ (all in the last bits of a computed
//! coordinate or distance). Of its 39 608 membership and relationship
//! answers, the 16 604 relationships are all unchanged, and 4 memberships
//! differ -- each for a probe point whose coordinates themselves come out
//! NaN with the intrinsics; every probe built the same both ways got the
//! same answer.
//!
//! # What Rust changes
//!
//! - Java's interfaces are traits ([`shape`]); abstract base classes'
//!   behaviour is in free functions each shape calls.
//! - Exceptions are [`Error`]; methods returning `null` return `Option`.
//! - `Plane extends Vector`, `SidedPlane extends Plane` and `GeoPoint extends
//!   Vector` become `Deref`. Where Java's dispatch depends on a `GeoPoint`'s
//!   cached magnitude, the method takes a `GeoPoint` (see
//!   [`geo_point::GeoPoint::arc_distance`]).
//! - Shapes are not `PartialEq` (Java's `equals` is only used by
//!   `equals`/`hashCode` themselves); serialized bytes are compared instead.

#![forbid(unsafe_code)]
// Java's range checks (`lat > PI * 0.5 || lat < -PI * 0.5`) let NaN through;
// `RangeInclusive::contains` would reject it, so they stay as written.
#![allow(clippy::manual_range_contains)]

pub mod bounds;
pub mod distance_style;
pub(crate) mod either_bound;
pub mod errors;
pub mod geo_area_factory;
pub mod geo_bbox_factory;
pub mod geo_circle_factory;
pub mod geo_complex_polygon;
pub mod geo_composite;
pub mod geo_convex_polygon;
pub mod geo_degenerate_horizontal_line;
pub mod geo_degenerate_path;
pub mod geo_degenerate_point;
pub mod geo_degenerate_vertical_line;
pub mod geo_exact_circle;
pub mod geo_latitude_zone;
pub mod geo_longitude_slice;
pub mod geo_north_rectangle;
pub mod geo_path_factory;
pub mod geo_point;
pub mod geo_polygon_factory;
pub mod geo_rectangle;
pub mod geo_s2_shape;
pub mod geo_south_rectangle;
pub mod geo_standard_circle;
pub mod geo_standard_path;
pub mod geo_wide_north_rectangle;
pub mod geo_wide_rectangle;
pub mod geo_wide_south_rectangle;
pub mod geo_world;
pub(crate) mod jmath;
pub mod lat_lon_bounds;
pub mod membership;
pub mod plane;
pub mod planet_model;
pub(crate) mod prelude;
pub mod serializable;
pub mod shape;
pub mod sided_plane;
pub mod standard_objects;
pub mod tools;
pub mod vector;
pub mod xyz_bounds;
pub mod xyz_solid;

pub use bounds::{Bounded, Bounds};
pub use distance_style::DistanceStyle;
pub use geo_point::GeoPoint;
pub use jmath::{to_degrees, to_radians};
pub use lat_lon_bounds::LatLonBounds;
pub use membership::Membership;
pub use plane::Plane;
pub use planet_model::{DocValueEncoder, PlanetModel};
pub use shape::*;
pub use sided_plane::SidedPlane;
pub use vector::Vector;
pub use xyz_bounds::XYZBounds;

/// The exceptions geo3d throws.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    /// `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// `IllegalStateException`.
    #[error("{0}")]
    IllegalState(String),
    /// `IOException` from deserialization.
    #[error("{0}")]
    Io(String),
    /// `RuntimeException` (thrown as itself, e.g. by `GeoStandardPath`).
    #[error("{0}")]
    Runtime(String),
    /// `NullPointerException`: where Java dereferences a `null` geo3d
    /// returned (a plane that could not be built, an unset bound). The
    /// message says what was missing; Java's is the JVM's own.
    #[error("{0}")]
    NullPointer(String),
    /// `IndexOutOfBoundsException`: a `List.get` past the end (an empty
    /// point list).
    #[error("{0}")]
    IndexOutOfBounds(String),
    /// `ArrayIndexOutOfBoundsException`: the same on an array.
    #[error("{0}")]
    ArrayIndexOutOfBounds(String),
}

impl Error {
    /// The Java exception class this stands for.
    pub fn java_class(&self) -> &'static str {
        match self {
            Error::IllegalArgument(_) => "java.lang.IllegalArgumentException",
            Error::IllegalState(_) => "java.lang.IllegalStateException",
            Error::Io(_) => "java.io.IOException",
            Error::Runtime(_) => "java.lang.RuntimeException",
            Error::NullPointer(_) => "java.lang.NullPointerException",
            Error::IndexOutOfBounds(_) => "java.lang.IndexOutOfBoundsException",
            Error::ArrayIndexOutOfBounds(_) => "java.lang.ArrayIndexOutOfBoundsException",
        }
    }
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

#[cfg(test)]
mod tests;
