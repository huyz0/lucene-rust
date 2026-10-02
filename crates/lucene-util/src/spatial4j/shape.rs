//! Spatial4j's shape interfaces -- `Shape`, `Point`, `Rectangle`, `Circle`
//! -- and `SpatialRelation`.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::Result;

/// `SpatialRelation`: how one shape relates to another. `WITHIN` and
/// `CONTAINS` include their boundaries ("covered by"/"covers").
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpatialRelation {
    Within,
    Contains,
    Disjoint,
    Intersects,
}

impl SpatialRelation {
    /// `transpose()`: the relation from the other shape's point of view.
    pub fn transpose(self) -> SpatialRelation {
        match self {
            SpatialRelation::Contains => SpatialRelation::Within,
            SpatialRelation::Within => SpatialRelation::Contains,
            other => other,
        }
    }

    /// `combine(other)`: the relation of a shape to the union of two
    /// shapes this one relates to as `self` and `other`. `None` is Java's
    /// `null` (the identity).
    pub fn combine(self, other: Option<SpatialRelation>) -> SpatialRelation {
        let Some(other) = other else {
            return self;
        };
        if other == self {
            return self;
        }
        if self == SpatialRelation::Disjoint && other == SpatialRelation::Contains
            || self == SpatialRelation::Contains && other == SpatialRelation::Disjoint
        {
            return SpatialRelation::Contains;
        }
        SpatialRelation::Intersects
    }

    /// `intersects()`: anything but `DISJOINT`.
    pub fn intersects(self) -> bool {
        self != SpatialRelation::Disjoint
    }

    /// `inverse()`: the relation to the shape's complement.
    pub fn inverse(self) -> SpatialRelation {
        match self {
            SpatialRelation::Disjoint => SpatialRelation::Contains,
            SpatialRelation::Contains => SpatialRelation::Disjoint,
            SpatialRelation::Within | SpatialRelation::Intersects => SpatialRelation::Intersects,
        }
    }

    /// The enum constant's name, as Java prints it.
    pub fn name(self) -> &'static str {
        match self {
            SpatialRelation::Within => "WITHIN",
            SpatialRelation::Contains => "CONTAINS",
            SpatialRelation::Disjoint => "DISJOINT",
            SpatialRelation::Intersects => "INTERSECTS",
        }
    }
}

impl fmt::Display for SpatialRelation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// `Shape`. `Display` is Java's `toString()`.
pub trait Shape: fmt::Debug + fmt::Display + Send + Sync + Any {
    /// `relate(other)`: how this shape relates to `other`. Fails where Java
    /// throws (a pair of shape types no implementation relates).
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation>;

    /// `getBoundingBox()`.
    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>>;

    /// `hasArea()`.
    fn has_area(&self) -> bool;

    /// `getArea(ctx)`: in the context's units squared, or (`None`) in
    /// degrees squared.
    fn area(&self, ctx: Option<&SpatialContext>) -> Result<f64>;

    /// `getCenter()`.
    fn center(&self) -> Result<Arc<dyn Point>>;

    /// `getBuffered(distance, ctx)`.
    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>>;

    /// `isEmpty()`.
    fn is_empty(&self) -> bool;

    /// `equals(other)`.
    fn equals(&self, other: &dyn Shape) -> bool;

    /// `getContext()`; `None` for the scratch points `BufferedLine` makes
    /// without one.
    fn context(&self) -> Option<&Arc<SpatialContext>>;

    /// For downcasting to the concrete type (Java's `instanceof` on a class).
    fn as_any(&self) -> &dyn Any;

    /// `this instanceof Point`.
    fn as_point(&self) -> Option<&dyn Point> {
        None
    }

    /// `this instanceof Rectangle`.
    fn as_rectangle(&self) -> Option<&dyn Rectangle> {
        None
    }

    /// `this instanceof Circle`.
    fn as_circle(&self) -> Option<&dyn Circle> {
        None
    }
}

/// `Point`: a location in x/y (longitude/latitude for geo).
pub trait Point: Shape {
    /// `getX()`: the longitude for geo.
    fn x(&self) -> f64;
    /// `getY()`: the latitude for geo.
    fn y(&self) -> f64;
    /// `getLat()`.
    fn lat(&self) -> f64 {
        self.y()
    }
    /// `getLon()`.
    fn lon(&self) -> f64 {
        self.x()
    }
}

/// `Rectangle`: an x/y range; for geo the x range may cross the dateline
/// (`min_x > max_x`).
pub trait Rectangle: Shape {
    fn min_x(&self) -> f64;
    fn max_x(&self) -> f64;
    fn min_y(&self) -> f64;
    fn max_y(&self) -> f64;
    /// `getWidth()`.
    fn width(&self) -> f64;
    /// `getHeight()`.
    fn height(&self) -> f64;
    /// `getCrossesDateLine()`.
    fn crosses_date_line(&self) -> bool;
    /// `relateYRange(minY, maxY)`.
    fn relate_y_range(&self, min_y: f64, max_y: f64) -> Result<SpatialRelation>;
    /// `relateXRange(minX, maxX)`.
    fn relate_x_range(&self, min_x: f64, max_x: f64) -> Result<SpatialRelation>;
}

/// `Circle`: a point and a radius (degrees for geo).
pub trait Circle: Shape {
    /// `getRadius()`.
    fn radius(&self) -> f64;
}

/// `Point.equals` (`PointImpl.equals(Point, Object)`): any point with the
/// same x and y (`Double.compare`).
pub fn point_equals(thiz: &dyn Point, o: &dyn Shape) -> bool {
    match o.as_point() {
        Some(p) => {
            super::double_compare_eq(p.x(), thiz.x()) && super::double_compare_eq(p.y(), thiz.y())
        }
        None => false,
    }
}

/// `RectangleImpl.equals(Rectangle, Object)`: any rectangle with the same
/// bounds. (Java casts the other rectangle to `RectangleImpl`, which throws
/// `ClassCastException` for a Geo3D rectangle; here the bounds are compared
/// for any rectangle.)
pub fn rectangle_equals(thiz: &dyn Rectangle, o: &dyn Shape) -> bool {
    match o.as_rectangle() {
        Some(r) => {
            super::double_compare_eq(r.max_x(), thiz.max_x())
                && super::double_compare_eq(r.max_y(), thiz.max_y())
                && super::double_compare_eq(r.min_x(), thiz.min_x())
                && super::double_compare_eq(r.min_y(), thiz.min_y())
        }
        None => false,
    }
}
