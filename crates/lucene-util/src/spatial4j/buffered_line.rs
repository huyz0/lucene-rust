//! `InfBufLine`, `BufferedLine` and `BufferedLineString`
//! (`org.locationtech.spatial4j.shape.impl`): planar line segments with a
//! buffer distance around them.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::collection::ShapeCollection;
use super::context::SpatialContext;
use super::distance::DistanceUtils;
use super::shape::{Point, Rectangle, Shape, SpatialRelation};
use super::{double_compare_eq, dstr, Error, Result};
use crate::geo::{java_max, java_min};

/// `InfBufLine`: an infinite line `y = slope * x + intercept` (vertical
/// when the slope is infinite, the intercept then on x) with a buffer.
#[derive(Debug, Clone, Copy)]
pub struct InfBufLine {
    slope: f64,
    intercept: f64,
    buf: f64,
    /// `1 / Math.sqrt(slope * slope + 1)`.
    dist_denom_inv: f64,
}

/// `InfBufLine.EPS`.
const EPS: f64 = 10e-14;

/// `oppositeQuad`: quadrants 1-4 are NE, NW, SW, SE.
const OPPOSITE_QUAD: [usize; 5] = [0, 3, 4, 1, 2];

impl InfBufLine {
    /// `new InfBufLine(slope, point, buf)`.
    pub fn new(slope: f64, point_x: f64, point_y: f64, buf: f64) -> Self {
        let (intercept, dist_denom_inv) = if slope.is_infinite() {
            (point_x, f64::NAN)
        } else {
            (
                point_y - slope * point_x,
                1.0 / (slope * slope + 1.0).sqrt(),
            )
        };
        InfBufLine {
            slope,
            intercept,
            buf,
            dist_denom_inv,
        }
    }

    /// `relate(r, prC, scratch)`: `pr_c` is `r`'s center.
    pub fn relate(&self, r: &dyn Rectangle, pr_c: (f64, f64)) -> SpatialRelation {
        let c_quad = self.quadrant(pr_c.0, pr_c.1);
        let nearest = Self::corner_by_quadrant(r, OPPOSITE_QUAD[c_quad]);
        if self.contains(nearest.0, nearest.1) {
            let farthest = Self::corner_by_quadrant(r, c_quad);
            if self.contains(farthest.0, farthest.1) {
                return SpatialRelation::Contains;
            }
            SpatialRelation::Intersects
        } else if self.quadrant(nearest.0, nearest.1) == c_quad {
            // out of buffer on same side as center
            SpatialRelation::Disjoint
        } else {
            // nearest & farthest points straddle the line
            SpatialRelation::Intersects
        }
    }

    /// `contains(p)`.
    pub fn contains(&self, x: f64, y: f64) -> bool {
        self.distance_unbuffered(x, y) <= self.buf + EPS
    }

    /// `distanceUnbuffered(c)`.
    pub fn distance_unbuffered(&self, x: f64, y: f64) -> f64 {
        if self.slope.is_infinite() {
            return (x - self.intercept).abs();
        }
        let num = (y - self.slope * x - self.intercept).abs();
        num * self.dist_denom_inv
    }

    /// `quadrant(c)`: 1-4 (NE, NW, SW, SE) relative to the line.
    pub fn quadrant(&self, x: f64, y: f64) -> usize {
        if self.slope.is_infinite() {
            return if x > self.intercept { 1 } else { 2 };
        }
        let y_at_c_in_line = self.slope * x + self.intercept;
        let above = y >= y_at_c_in_line;
        if self.slope > 0.0 {
            if above {
                2
            } else {
                4
            }
        } else if above {
            1
        } else {
            3
        }
    }

    /// `cornerByQuadrant(r, cornerQuad, out)`.
    pub fn corner_by_quadrant(r: &dyn Rectangle, corner_quad: usize) -> (f64, f64) {
        let x = if corner_quad == 1 || corner_quad == 4 {
            r.max_x()
        } else {
            r.min_x()
        };
        let y = if corner_quad == 1 || corner_quad == 2 {
            r.max_y()
        } else {
            r.min_y()
        };
        (x, y)
    }

    /// `getSlope()`.
    pub fn slope(&self) -> f64 {
        self.slope
    }

    /// `getIntercept()`.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// `getBuf()`.
    pub fn buf(&self) -> f64 {
        self.buf
    }

    /// `getDistDenomInv()`.
    pub fn dist_denom_inv(&self) -> f64 {
        self.dist_denom_inv
    }
}

impl fmt::Display for InfBufLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InfBufLine{{buf={}, intercept={}, slope={}}}",
            dstr(self.buf),
            dstr(self.intercept),
            dstr(self.slope)
        )
    }
}

/// `BufferedLine`: the segment `a`-`b` buffered by `buf` (the buffer
/// extends past the ends too).
#[derive(Debug, Clone)]
pub struct BufferedLine {
    p_a: Arc<dyn Point>,
    p_b: Arc<dyn Point>,
    buf: f64,
    bbox: Arc<dyn Rectangle>,
    line_primary: InfBufLine,
    line_perp: InfBufLine,
    ctx: Arc<SpatialContext>,
}

impl BufferedLine {
    /// `new BufferedLine(pA, pB, buf, ctx)`.
    pub fn new(
        p_a: Arc<dyn Point>,
        p_b: Arc<dyn Point>,
        buf: f64,
        ctx: Arc<SpatialContext>,
    ) -> Result<Self> {
        let delta_y = p_b.y() - p_a.y();
        let delta_x = p_b.x() - p_a.x();
        let center = (p_a.x() + delta_x / 2.0, p_a.y() + delta_y / 2.0);
        let perp_extent = buf;
        let (line_primary, line_perp);
        if delta_x == 0.0 && delta_y == 0.0 {
            line_primary = InfBufLine::new(0.0, center.0, center.1, buf);
            line_perp = InfBufLine::new(f64::INFINITY, center.0, center.1, buf);
        } else {
            line_primary = InfBufLine::new(delta_y / delta_x, center.0, center.1, buf);
            let length = (delta_x * delta_x + delta_y * delta_y).sqrt();
            line_perp = InfBufLine::new(
                -delta_x / delta_y,
                center.0,
                center.1,
                length / 2.0 + perp_extent,
            );
        }
        let (mut min_y, mut max_y, min_x, max_x);
        if delta_x == 0.0 {
            // vertical
            if p_a.y() <= p_b.y() {
                min_y = p_a.y();
                max_y = p_b.y();
            } else {
                min_y = p_b.y();
                max_y = p_a.y();
            }
            min_x = p_a.x() - buf;
            max_x = p_a.x() + buf;
            min_y -= perp_extent;
            max_y += perp_extent;
        } else {
            let bbox_buf = buf * (1.0 + line_primary.slope().abs()) * line_primary.dist_denom_inv();
            if p_a.x() <= p_b.x() {
                min_x = p_a.x() - bbox_buf;
                max_x = p_b.x() + bbox_buf;
            } else {
                min_x = p_b.x() - bbox_buf;
                max_x = p_a.x() + bbox_buf;
            }
            if p_a.y() <= p_b.y() {
                min_y = p_a.y() - bbox_buf;
                max_y = p_b.y() + bbox_buf;
            } else {
                min_y = p_b.y() - bbox_buf;
                max_y = p_a.y() + bbox_buf;
            }
        }
        let [bminx, bmaxx, bminy, bmaxy] = ctx.world_bounds_values();
        let bbox = ctx.rect(
            java_max(bminx, min_x),
            java_min(bmaxx, max_x),
            java_max(bminy, min_y),
            java_min(bmaxy, max_y),
        )?;
        Ok(BufferedLine {
            p_a,
            p_b,
            buf,
            bbox,
            line_primary,
            line_perp,
            ctx,
        })
    }

    /// `expandBufForLongitudeSkew(pA, pB, buf)`.
    pub fn expand_buf_for_longitude_skew(p_a: &dyn Point, p_b: &dyn Point, buf: f64) -> f64 {
        let abs_a = p_a.y().abs();
        let abs_b = p_b.y().abs();
        let max_lat = java_max(abs_a, abs_b);
        DistanceUtils::calc_lon_degrees_at_lat(max_lat, buf)
    }

    /// `relate(Rectangle)`.
    pub fn relate_rect(&self, r: &dyn Rectangle) -> Result<SpatialRelation> {
        let bbox_r = self.bbox.relate(r)?;
        if bbox_r == SpatialRelation::Disjoint || bbox_r == SpatialRelation::Within {
            return Ok(bbox_r);
        }
        let pr_c = r.center()?;
        let pr_c = (pr_c.x(), pr_c.y());
        let result = self.line_primary.relate(r, pr_c);
        if result == SpatialRelation::Disjoint {
            return Ok(SpatialRelation::Disjoint);
        }
        let result_opp = self.line_perp.relate(r, pr_c);
        if result_opp == SpatialRelation::Disjoint {
            return Ok(SpatialRelation::Disjoint);
        }
        if result == result_opp {
            return Ok(result);
        }
        Ok(SpatialRelation::Intersects)
    }

    /// `contains(p)`.
    pub fn contains(&self, p: &dyn Point) -> bool {
        self.line_primary.contains(p.x(), p.y()) && self.line_perp.contains(p.x(), p.y())
    }

    /// `getA()`.
    pub fn a(&self) -> &Arc<dyn Point> {
        &self.p_a
    }

    /// `getB()`.
    pub fn b(&self) -> &Arc<dyn Point> {
        &self.p_b
    }

    /// `getBuf()`.
    pub fn buf(&self) -> f64 {
        self.buf
    }

    /// `getLinePrimary()`.
    pub fn line_primary(&self) -> &InfBufLine {
        &self.line_primary
    }

    /// `getLinePerp()`.
    pub fn line_perp(&self) -> &InfBufLine {
        &self.line_perp
    }
}

impl fmt::Display for BufferedLine {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BufferedLine({}, {} b={})",
            self.p_a,
            self.p_b,
            dstr(self.buf)
        )
    }
}

impl Shape for BufferedLine {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if let Some(p) = other.as_point() {
            return Ok(if self.contains(p) {
                SpatialRelation::Contains
            } else {
                SpatialRelation::Disjoint
            });
        }
        if let Some(r) = other.as_rectangle() {
            return self.relate_rect(r);
        }
        Err(Error::UnsupportedOperation(None))
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        Ok(self.bbox.clone())
    }

    fn has_area(&self) -> bool {
        self.buf > 0.0
    }

    fn area(&self, _ctx: Option<&SpatialContext>) -> Result<f64> {
        Ok(self.line_primary.buf() * self.line_perp.buf() * 4.0)
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        self.bbox.center()
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(BufferedLine::new(
            self.p_a.clone(),
            self.p_b.clone(),
            self.buf + distance,
            ctx.clone(),
        )?))
    }

    fn is_empty(&self) -> bool {
        self.p_a.is_empty()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        let Some(that) = other.as_any().downcast_ref::<BufferedLine>() else {
            return false;
        };
        double_compare_eq(that.buf, self.buf)
            && self.p_a.equals(&*that.p_a)
            && self.p_b.equals(&*that.p_b)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `BufferedLineString`: consecutive [`BufferedLine`] segments.
#[derive(Debug, Clone)]
pub struct BufferedLineString {
    segments: Vec<Arc<BufferedLine>>,
    collection: ShapeCollection,
    buf: f64,
    ctx: Arc<SpatialContext>,
}

impl BufferedLineString {
    /// `new BufferedLineString(points, buf, expandBufForLongitudeSkew, ctx)`.
    pub fn new(
        points: &[Arc<dyn Point>],
        buf: f64,
        expand_buf_for_longitude_skew: bool,
        ctx: Arc<SpatialContext>,
    ) -> Result<Self> {
        let mut segments: Vec<Arc<BufferedLine>> = Vec::new();
        if !points.is_empty() {
            let mut prev: Option<&Arc<dyn Point>> = None;
            for point in points {
                if let Some(prev_point) = prev {
                    let seg_buf = if expand_buf_for_longitude_skew {
                        BufferedLine::expand_buf_for_longitude_skew(&**prev_point, &**point, buf)
                    } else {
                        buf
                    };
                    segments.push(Arc::new(BufferedLine::new(
                        prev_point.clone(),
                        point.clone(),
                        seg_buf,
                        ctx.clone(),
                    )?));
                }
                prev = Some(point);
            }
            if segments.is_empty() {
                let p = prev.expect("points is not empty").clone();
                segments.push(Arc::new(BufferedLine::new(p.clone(), p, buf, ctx.clone())?));
            }
        }
        let collection = ctx.collection(
            segments
                .iter()
                .map(|s| s.clone() as Arc<dyn Shape>)
                .collect(),
        )?;
        Ok(BufferedLineString {
            segments,
            collection,
            buf,
            ctx,
        })
    }

    /// `getSegments()`.
    pub fn segments(&self) -> &[Arc<BufferedLine>] {
        &self.segments
    }

    /// `getBuf()`.
    pub fn buf(&self) -> f64 {
        self.buf
    }

    /// `getPoints()`.
    pub fn points(&self) -> Vec<Arc<dyn Point>> {
        let Some(first) = self.segments.first() else {
            return Vec::new();
        };
        let mut pts = vec![first.a().clone()];
        pts.extend(self.segments.iter().map(|s| s.b().clone()));
        pts
    }
}

impl fmt::Display for BufferedLineString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferedLineString(buf={} pts=", dstr(self.buf))?;
        for (i, point) in self.points().iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} {}", dstr(point.x()), dstr(point.y()))?;
        }
        f.write_str(")")
    }
}

impl Shape for BufferedLineString {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        self.collection.relate(other)
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        self.collection.bounding_box()
    }

    fn has_area(&self) -> bool {
        self.collection.has_area()
    }

    fn area(&self, ctx: Option<&SpatialContext>) -> Result<f64> {
        self.collection.area(ctx)
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        self.collection.center()
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        ctx.line_string(&self.points(), self.buf + distance)
    }

    fn is_empty(&self) -> bool {
        self.segments.is_empty()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        let Some(that) = other.as_any().downcast_ref::<BufferedLineString>() else {
            return false;
        };
        double_compare_eq(that.buf, self.buf) && self.collection.equals(&that.collection)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
