//! `PointImpl` (`org.locationtech.spatial4j.shape.impl.PointImpl`).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::shape::{point_equals, Point, Rectangle, Shape, SpatialRelation};
use super::{dstr, Error, Result};

/// `PointImpl`: a point; empty when `x` is NaN.
#[derive(Debug, Clone)]
pub struct PointImpl {
    x: f64,
    y: f64,
    ctx: Option<Arc<SpatialContext>>,
}

impl PointImpl {
    /// `new PointImpl(x, y, ctx)`: no validation.
    pub fn new(x: f64, y: f64, ctx: Arc<SpatialContext>) -> Self {
        PointImpl {
            x,
            y,
            ctx: Some(ctx),
        }
    }

    /// `new PointImpl(x, y, null)`: the context-less scratch point
    /// `BufferedLine` uses.
    pub fn without_context(x: f64, y: f64) -> Self {
        PointImpl { x, y, ctx: None }
    }

    fn ctx(&self) -> Result<&Arc<SpatialContext>> {
        self.ctx
            .as_ref()
            .ok_or_else(|| Error::NullPointer("null".into()))
    }
}

impl fmt::Display for PointImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pt(x={},y={})", dstr(self.x), dstr(self.y))
    }
}

impl Shape for PointImpl {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if self.is_empty() || other.is_empty() {
            return Ok(SpatialRelation::Disjoint);
        }
        if other.as_point().is_some() {
            return Ok(if self.equals(other) {
                SpatialRelation::Intersects
            } else {
                SpatialRelation::Disjoint
            });
        }
        Ok(other.relate(self)?.transpose())
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        self.ctx()?.rect(self.x, self.x, self.y, self.y)
    }

    fn has_area(&self) -> bool {
        false
    }

    fn area(&self, _ctx: Option<&SpatialContext>) -> Result<f64> {
        Ok(0.0)
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        Ok(Arc::new(self.clone()))
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        Ok(ctx.circle_at(&(Arc::new(self.clone()) as Arc<dyn Point>), distance)?)
    }

    fn is_empty(&self) -> bool {
        self.x.is_nan()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        point_equals(self, other)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        self.ctx.as_ref()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_point(&self) -> Option<&dyn Point> {
        Some(self)
    }
}

impl Point for PointImpl {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}
