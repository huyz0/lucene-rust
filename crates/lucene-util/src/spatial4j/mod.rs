//! Port of the subset of [Spatial4j](https://github.com/locationtech/spatial4j)
//! 0.8 (`org.locationtech.spatial4j`, Apache License 2.0) that Lucene
//! 10.5.0's `lucene-spatial-extras` exercises: the spatial context and its
//! factory, the shapes (point, rectangle, circle -- planar and geodetic --,
//! shape collection, buffered line and line string), their relations,
//! bounding boxes, buffers and areas, the distance calculators and
//! `DistanceUtils`, the WKT reader, `BinaryCodec` and `GeohashUtils`.
//!
//! Spatial4j is a third-party Java library with no Rust equivalent; Lucene's
//! spatial-extras is built on its shape model, so the strategies and prefix
//! trees cannot be ported without it. It is pure geometry with no index
//! dependency, so it lives in `lucene-util` beside `geo` and `spatial3d`.
//! `crates/lucene-util/tests/spatial4j_fixtures.rs` compares it against the
//! real Spatial4j 0.8 jar (`fixtures/src/GenSpatial4j.java`).
//!
//! # Left out, and why
//!
//! - JTS (`org.locationtech.spatial4j.*.jts`): an optional dependency of
//!   Spatial4j that Lucene does not ship; without it a polygon is only
//!   available through Lucene's Geo3D context
//!   (`crate::spatial_extras::spatial4j`), as in Java.
//! - The GeoJSON, Polyshape and legacy readers/writers, `WKTWriter`,
//!   `SupportedFormats` and the Jackson modules: spatial-extras only reads
//!   WKT (`SpatialArgsParser`) and writes the binary codec
//!   (`SerializedDVStrategy`). GeoJSON is not even registered in Java unless
//!   the optional noggit jar is on the class path, which Lucene's is not.
//! - `Range` (deprecated, superseded by [`bbox_calculator`]) and the
//!   deprecated `DistanceUtils.vector*` helpers.
//! - `Shape.hashCode`: nothing spatial-extras ports hashes a shape.
//!
//! # What Rust changes
//!
//! - Java's interfaces are traits ([`Shape`], [`Point`], [`Rectangle`],
//!   [`Circle`], [`DistanceCalculator`], [`ShapeFactory`],
//!   [`BinaryCodec`]); shapes are shared as `Arc<dyn ...>`, and `instanceof`
//!   is [`Shape::as_point`]/[`Shape::as_rectangle`]/[`Shape::as_circle`] or
//!   a downcast through [`Shape::as_any`].
//! - A shape holds its context as an `Arc`; the context does not hold its
//!   world bounds or factory's back-reference as shapes (no cycle), so a
//!   factory method takes the context as an argument
//!   ([`SpatialContext::point_xy`] and friends are the convenient form).
//! - Java's mutable `reset` methods and the "reuse" arguments are not
//!   ported: every result is a fresh shape (`reset` only exists to avoid
//!   garbage).
//! - Exceptions are [`Error`], with Java's message and class
//!   ([`Error::java_class`]).
//! - `Math.sin`/`cos` are `StrictMath`'s fdlibm ([`crate::strict_math`]), as
//!   in `spatial3d`: the fixtures are generated with HotSpot's trig
//!   intrinsics off, so the comparison is bit for bit.

#![forbid(unsafe_code)]
// Java's range checks let NaN through; `RangeInclusive::contains` would not.
#![allow(clippy::manual_range_contains)]

pub mod bbox_calculator;
pub mod binary_codec;
pub mod buffered_line;
pub mod circle;
pub mod collection;
pub mod context;
pub mod distance;
pub mod geohash;
pub mod point;
pub mod rectangle;
pub mod shape;
pub mod shape_factory;
pub mod wkt;

pub use bbox_calculator::BBoxCalculator;
pub use binary_codec::{BinaryCodec, DefaultBinaryCodec};
pub use buffered_line::{BufferedLine, BufferedLineString, InfBufLine};
pub use circle::CircleImpl;
pub use collection::ShapeCollection;
pub use context::{SpatialContext, SpatialContextFactory};
pub use distance::{CartesianDistCalc, DistanceCalculator, DistanceUtils, GeodesicSphereDistCalc};
pub use point::PointImpl;
pub use rectangle::RectangleImpl;
pub use shape::{Circle, Point, Rectangle, Shape, SpatialRelation};
pub use shape_factory::{
    LineStringBuilder, MultiLineStringBuilder, MultiPointBuilder, MultiPolygonBuilder,
    MultiShapeBuilder, PolygonBuilder, ShapeFactory, ShapeFactoryImpl,
};
pub use wkt::WktReader;

/// The exceptions Spatial4j (and the Lucene classes plugged into it) throw.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// `org.locationtech.spatial4j.exception.InvalidShapeException`.
    #[error("{0}")]
    InvalidShape(String),
    /// `java.text.ParseException`, with its error offset.
    #[error("{message}")]
    Parse { message: String, offset: i32 },
    /// `IllegalArgumentException`.
    #[error("{0}")]
    IllegalArgument(String),
    /// `UnsupportedOperationException`; `None` for one thrown without a
    /// message.
    #[error("{}", .0.as_deref().unwrap_or("null"))]
    UnsupportedOperation(Option<String>),
    /// `RuntimeException` (thrown as itself).
    #[error("{0}")]
    Runtime(String),
    /// `ClassCastException` (a Geo3D builder handed a non-Geo3D shape).
    #[error("{0}")]
    ClassCast(String),
    /// `java.io.IOException` / `EOFException` while reading a binary shape.
    #[error("{0}")]
    Io(String),
    /// `NullPointerException` (Java dereferencing an unset value).
    #[error("{0}")]
    NullPointer(String),
    /// `ArrayIndexOutOfBoundsException`.
    #[error("{0}")]
    ArrayIndexOutOfBounds(String),
    /// `NumberFormatException` (`Double.valueOf`, `Integer.valueOf`).
    #[error("{0}")]
    NumberFormat(String),
    /// A geo3d (`spatial3d`) exception, passed through.
    #[error("{0}")]
    Spatial3d(crate::spatial3d::Error),
}

impl Error {
    /// The Java exception class this stands for.
    pub fn java_class(&self) -> &'static str {
        match self {
            Error::InvalidShape(_) => "org.locationtech.spatial4j.exception.InvalidShapeException",
            Error::Parse { .. } => "java.text.ParseException",
            Error::IllegalArgument(_) => "java.lang.IllegalArgumentException",
            Error::UnsupportedOperation(_) => "java.lang.UnsupportedOperationException",
            Error::Runtime(_) => "java.lang.RuntimeException",
            Error::ClassCast(_) => "java.lang.ClassCastException",
            Error::Io(_) => "java.io.IOException",
            Error::NullPointer(_) => "java.lang.NullPointerException",
            Error::ArrayIndexOutOfBounds(_) => "java.lang.ArrayIndexOutOfBoundsException",
            Error::NumberFormat(_) => "java.lang.NumberFormatException",
            Error::Spatial3d(e) => e.java_class(),
        }
    }

    /// `Throwable.toString()`: the class name, then `": "` and the message
    /// when there is one.
    pub fn java_to_string(&self) -> String {
        match self {
            Error::UnsupportedOperation(None) => self.java_class().to_string(),
            _ => format!("{}: {}", self.java_class(), self),
        }
    }
}

impl From<crate::spatial3d::Error> for Error {
    fn from(e: crate::spatial3d::Error) -> Self {
        Error::Spatial3d(e)
    }
}

/// `Result` with [`Error`].
pub type Result<T> = std::result::Result<T, Error>;

/// `Double.compare(a, b) == 0`: equal bits after NaN canonicalisation, so
/// `0.0 != -0.0` and `NaN == NaN`.
#[inline]
pub(crate) fn double_compare_eq(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    a.to_bits() == b.to_bits()
}

/// `Double.toString(v)`.
#[inline]
pub(crate) fn dstr(v: f64) -> String {
    crate::geo::java_double_string(v)
}

/// `String.format(Locale.ROOT, "%.<prec>f", v)`: Java rounds the
/// shortest decimal representation half-up (not the exact binary value
/// half-even, as Rust's `{:.prec$}` does).
pub(crate) fn java_format_fixed(v: f64, prec: usize) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Infinity" } else { "-Infinity" }.into();
    }
    let sci = format!("{:e}", v.abs());
    let (mantissa, exp) = sci.split_once('e').expect("{:e} has an exponent");
    let exp: i64 = exp.parse().expect("{:e}'s exponent is an integer");
    let digits: Vec<u8> = mantissa
        .bytes()
        .filter(u8::is_ascii_digit)
        .map(|b| b - b'0')
        .collect();
    // digits[i] has weight 10^(exp - i).
    let digit_at = |w: i64| -> u8 {
        let i = exp - w;
        if i >= 0 && (i as usize) < digits.len() {
            digits[i as usize]
        } else {
            0
        }
    };
    let top = exp.max(0);
    let prec_w = prec as i64;
    let mut kept: Vec<u8> = (-prec_w..=top).rev().map(digit_at).collect();
    let next = digit_at(-prec_w - 1);
    if next >= 5 {
        let mut i = kept.len();
        loop {
            if i == 0 {
                kept.insert(0, 1);
                break;
            }
            i -= 1;
            if kept[i] == 9 {
                kept[i] = 0;
            } else {
                kept[i] += 1;
                break;
            }
        }
    }
    let total = kept.len();
    let int_len = total.saturating_sub(prec);
    let mut s = String::new();
    if v.is_sign_negative() {
        s.push('-');
    }
    // `kept` holds at least one integer digit and exactly `prec` fraction
    // digits.
    s.extend(kept[..int_len].iter().map(|d| (b'0' + d) as char));
    if prec > 0 {
        s.push('.');
        s.extend(kept[int_len..].iter().map(|d| (b'0' + d) as char));
    }
    s
}

#[cfg(test)]
mod tests;
