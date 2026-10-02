//! `ShapeCollection` (`org.locationtech.spatial4j.shape.ShapeCollection`):
//! a list of shapes treated as one (OGC's GeometryCollection).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::bbox_calculator::BBoxCalculator;
use super::context::SpatialContext;
use super::shape::{Point, Rectangle, Shape, SpatialRelation};
use super::Result;

/// `ShapeCollection`.
#[derive(Debug, Clone)]
pub struct ShapeCollection {
    shapes: Vec<Arc<dyn Shape>>,
    ctx: Arc<SpatialContext>,
    bbox: Arc<dyn Rectangle>,
}

impl ShapeCollection {
    /// `new ShapeCollection(shapes, ctx)`.
    pub fn new(shapes: Vec<Arc<dyn Shape>>, ctx: Arc<SpatialContext>) -> Result<Self> {
        let bbox = Self::compute_bounding_box(&shapes, &ctx)?;
        Ok(ShapeCollection { shapes, ctx, bbox })
    }

    /// `computeBoundingBox(shapes, ctx)`.
    fn compute_bounding_box(
        shapes: &[Arc<dyn Shape>],
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Rectangle>> {
        if shapes.is_empty() {
            return ctx.rect(f64::NAN, f64::NAN, f64::NAN, f64::NAN);
        }
        let mut calc = BBoxCalculator::new(ctx.clone());
        for geom in shapes {
            calc.expand_range_rect(&*geom.bounding_box()?);
        }
        calc.boundary()
    }

    /// `getShapes()`.
    pub fn shapes(&self) -> &[Arc<dyn Shape>] {
        &self.shapes
    }

    /// `get(index)`.
    pub fn get(&self, index: usize) -> &Arc<dyn Shape> {
        &self.shapes[index]
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.shapes.len()
    }

    /// `relateContainsShortCircuits()`.
    fn relate_contains_short_circuits(&self) -> bool {
        true
    }
}

impl fmt::Display for ShapeCollection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut buf = String::from("ShapeCollection(");
        for (i, shape) in self.shapes.iter().enumerate() {
            if i > 0 {
                buf.push_str(", ");
            }
            buf.push_str(&shape.to_string());
            // Java's StringBuilder.length() counts UTF-16 units.
            if buf.encode_utf16().count() > 150 {
                buf.push_str(&format!(" ...{}", self.shapes.len()));
                break;
            }
        }
        buf.push(')');
        f.write_str(&buf)
    }
}

impl Shape for ShapeCollection {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        let bbox_sect = self.bbox.relate(other)?;
        if bbox_sect == SpatialRelation::Disjoint || bbox_sect == SpatialRelation::Within {
            return Ok(bbox_sect);
        }
        let contains_will_short_circuit =
            other.as_point().is_some() || self.relate_contains_short_circuits();
        let mut sect: Option<SpatialRelation> = None;
        for shape in &self.shapes {
            let next_sect = shape.relate(other)?;
            let s = match sect {
                None => next_sect,
                Some(s) => s.combine(Some(next_sect)),
            };
            sect = Some(s);
            if s == SpatialRelation::Intersects {
                return Ok(SpatialRelation::Intersects);
            }
            if s == SpatialRelation::Contains && contains_will_short_circuit {
                return Ok(SpatialRelation::Contains);
            }
        }
        // Java returns null for an empty collection whose bbox did not
        // already decide; an empty bbox is NaN, which relates DISJOINT.
        Ok(sect.unwrap_or(SpatialRelation::Disjoint))
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        Ok(self.bbox.clone())
    }

    fn has_area(&self) -> bool {
        self.shapes.iter().any(|s| s.has_area())
    }

    fn area(&self, ctx: Option<&SpatialContext>) -> Result<f64> {
        let max_area = self.bbox.area(ctx)?;
        let mut sum = 0.0;
        for geom in &self.shapes {
            sum += geom.area(ctx)?;
            if sum >= max_area {
                return Ok(max_area);
            }
        }
        Ok(sum)
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        self.bbox.center()
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        let mut buf = Vec::with_capacity(self.shapes.len());
        for shape in &self.shapes {
            buf.push(shape.buffered(distance, ctx)?);
        }
        Ok(Arc::new(ctx.collection(buf)?))
    }

    fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        let Some(that) = other.as_any().downcast_ref::<ShapeCollection>() else {
            return false;
        };
        self.shapes.len() == that.shapes.len()
            && self
                .shapes
                .iter()
                .zip(that.shapes.iter())
                .all(|(a, b)| a.equals(&**b))
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
