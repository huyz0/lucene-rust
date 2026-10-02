//! `RectangleImpl` (`org.locationtech.spatial4j.shape.impl.RectangleImpl`).

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::distance::DistanceUtils;
use super::point::PointImpl;
use super::shape::{rectangle_equals, Point, Rectangle, Shape, SpatialRelation};
use super::{dstr, Result};
use crate::geo::{java_max, java_min};

/// `RectangleImpl`: an x/y range; for geo, `min_x > max_x` crosses the
/// dateline. Empty when `min_x` is NaN.
#[derive(Debug, Clone)]
pub struct RectangleImpl {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    ctx: Arc<SpatialContext>,
}

impl RectangleImpl {
    /// `new RectangleImpl(minX, maxX, minY, maxY, ctx)`: no normalisation
    /// or validation.
    pub fn new(min_x: f64, max_x: f64, min_y: f64, max_y: f64, ctx: Arc<SpatialContext>) -> Self {
        RectangleImpl {
            min_x,
            max_x,
            min_y,
            max_y,
            ctx,
        }
    }

    /// `relate(Point)`.
    pub fn relate_point(&self, point: &dyn Point) -> SpatialRelation {
        if point.y() > self.max_y || point.y() < self.min_y {
            return SpatialRelation::Disjoint;
        }
        let min_x = self.min_x;
        let mut max_x = self.max_x;
        let mut p_x = point.x();
        if self.ctx.is_geo() {
            // unwrap dateline and normalize +180 to become -180
            let raw_width = max_x - min_x;
            if raw_width < 0.0 {
                max_x = min_x + (raw_width + 360.0);
            }
            // shift to potentially overlap
            if p_x < min_x {
                p_x += 360.0;
            } else if p_x > max_x {
                p_x -= 360.0;
            } else {
                return SpatialRelation::Contains;
            }
        }
        if p_x < min_x || p_x > max_x {
            return SpatialRelation::Disjoint;
        }
        SpatialRelation::Contains
    }

    /// `relate(Rectangle)`.
    pub fn relate_rect(&self, rect: &dyn Rectangle) -> Result<SpatialRelation> {
        let y_intersect = self.relate_y_range_impl(rect.min_y(), rect.max_y());
        if y_intersect == SpatialRelation::Disjoint {
            return Ok(SpatialRelation::Disjoint);
        }
        let x_intersect = self.relate_x_range_impl(rect.min_x(), rect.max_x());
        if x_intersect == SpatialRelation::Disjoint {
            return Ok(SpatialRelation::Disjoint);
        }
        if x_intersect == y_intersect {
            return Ok(x_intersect);
        }
        // if one side is equal, return the other
        if self.min_y == rect.min_y() && self.max_y == rect.max_y() {
            return Ok(x_intersect);
        }
        if self.min_x == rect.min_x() && self.max_x == rect.max_x()
            || (self.ctx.is_geo() && vertical_at_dateline(self, rect))
        {
            return Ok(y_intersect);
        }
        Ok(SpatialRelation::Intersects)
    }

    /// `relateYRange(ext_minY, ext_maxY)`.
    pub fn relate_y_range_impl(&self, ext_min_y: f64, ext_max_y: f64) -> SpatialRelation {
        relate_range(self.min_y, self.max_y, ext_min_y, ext_max_y)
    }

    /// `relateXRange(ext_minX, ext_maxX)`.
    pub fn relate_x_range_impl(&self, ext_min_x: f64, ext_max_x: f64) -> SpatialRelation {
        let mut min_x = self.min_x;
        let mut max_x = self.max_x;
        let mut ext_min_x = ext_min_x;
        let mut ext_max_x = ext_max_x;
        if self.ctx.is_geo() {
            // unwrap dateline, plus do world-wrap short circuit
            let raw_width = max_x - min_x;
            if raw_width == 360.0 {
                return SpatialRelation::Contains;
            }
            if raw_width < 0.0 {
                max_x = min_x + (raw_width + 360.0);
            }
            let ext_raw_width = ext_max_x - ext_min_x;
            if ext_raw_width == 360.0 {
                return SpatialRelation::Within;
            }
            if ext_raw_width < 0.0 {
                ext_max_x = ext_min_x + (ext_raw_width + 360.0);
            }
            // shift to potentially overlap
            if max_x < ext_min_x {
                min_x += 360.0;
                max_x += 360.0;
            } else if ext_max_x < min_x {
                ext_min_x += 360.0;
                ext_max_x += 360.0;
            }
        }
        relate_range(min_x, max_x, ext_min_x, ext_max_x)
    }

    /// `getCenter()` as the concrete point.
    pub fn center_point(&self) -> PointImpl {
        if self.min_x.is_nan() {
            return PointImpl::new(f64::NAN, f64::NAN, self.ctx.clone());
        }
        let y = self.height() / 2.0 + self.min_y;
        let mut x = self.width() / 2.0 + self.min_x;
        if self.min_x > self.max_x {
            x = DistanceUtils::norm_lon_deg(x);
        }
        PointImpl::new(x, y, self.ctx.clone())
    }
}

/// `verticalAtDateline(rect1, rect2)`.
fn vertical_at_dateline(rect1: &RectangleImpl, rect2: &dyn Rectangle) -> bool {
    if rect1.min_x == rect1.max_x && rect2.min_x() == rect2.max_x() {
        if rect1.min_x == -180.0 {
            return rect2.min_x() == 180.0;
        } else if rect1.min_x == 180.0 {
            return rect2.min_x() == -180.0;
        }
    }
    false
}

/// `relate_range(int_min, int_max, ext_min, ext_max)`.
fn relate_range(int_min: f64, int_max: f64, ext_min: f64, ext_max: f64) -> SpatialRelation {
    if ext_min > int_max || ext_max < int_min {
        return SpatialRelation::Disjoint;
    }
    if ext_min >= int_min && ext_max <= int_max {
        return SpatialRelation::Contains;
    }
    if ext_min <= int_min && ext_max >= int_max {
        return SpatialRelation::Within;
    }
    SpatialRelation::Intersects
}

impl fmt::Display for RectangleImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Rect(minX={},maxX={},minY={},maxY={})",
            dstr(self.min_x),
            dstr(self.max_x),
            dstr(self.min_y),
            dstr(self.max_y)
        )
    }
}

impl Shape for RectangleImpl {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if self.is_empty() || other.is_empty() {
            return Ok(SpatialRelation::Disjoint);
        }
        if let Some(p) = other.as_point() {
            return Ok(self.relate_point(p));
        }
        if let Some(r) = other.as_rectangle() {
            return self.relate_rect(r);
        }
        Ok(other.relate(self)?.transpose())
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        Ok(Arc::new(self.clone()))
    }

    fn has_area(&self) -> bool {
        self.max_x != self.min_x && self.max_y != self.min_y
    }

    fn area(&self, ctx: Option<&SpatialContext>) -> Result<f64> {
        match ctx {
            None => Ok(self.width() * self.height()),
            Some(ctx) => ctx.dist_calc().area_rect(self),
        }
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        Ok(Arc::new(self.center_point()))
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        let (min_x, max_x, min_y, max_y) = (self.min_x, self.max_x, self.min_y, self.max_y);
        let r = if ctx.is_geo() {
            // first check pole touching, triggering a world-wrap rect
            if max_y + distance >= 90.0 {
                ctx.rect(-180.0, 180.0, java_max(-90.0, min_y - distance), 90.0)?
            } else if min_y - distance <= -90.0 {
                ctx.rect(-180.0, 180.0, -90.0, java_min(90.0, max_y + distance))?
            } else {
                // doesn't touch pole
                let lat_distance = distance;
                let closest_to_pole_y = if max_y.abs() > min_y.abs() {
                    max_y
                } else {
                    min_y
                };
                let lon_distance = DistanceUtils::calc_box_by_dist_from_pt_delta_lon_deg(
                    closest_to_pole_y,
                    min_x,
                    distance,
                );
                // could still wrap the world though...
                if lon_distance * 2.0 + self.width() >= 360.0 {
                    ctx.rect(-180.0, 180.0, min_y - lat_distance, max_y + lat_distance)?
                } else {
                    ctx.rect(
                        DistanceUtils::norm_lon_deg(min_x - lon_distance),
                        DistanceUtils::norm_lon_deg(max_x + lon_distance),
                        min_y - lat_distance,
                        max_y + lat_distance,
                    )?
                }
            }
        } else {
            let [wminx, wmaxx, wminy, wmaxy] = ctx.world_bounds_values();
            let new_min_x = java_max(wminx, min_x - distance);
            let new_max_x = java_min(wmaxx, max_x + distance);
            let new_min_y = java_max(wminy, min_y - distance);
            let new_max_y = java_min(wmaxy, max_y + distance);
            ctx.rect(new_min_x, new_max_x, new_min_y, new_max_y)?
        };
        Ok(r)
    }

    fn is_empty(&self) -> bool {
        self.min_x.is_nan()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        rectangle_equals(self, other)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_rectangle(&self) -> Option<&dyn Rectangle> {
        Some(self)
    }
}

impl Rectangle for RectangleImpl {
    fn min_x(&self) -> f64 {
        self.min_x
    }

    fn max_x(&self) -> f64 {
        self.max_x
    }

    fn min_y(&self) -> f64 {
        self.min_y
    }

    fn max_y(&self) -> f64 {
        self.max_y
    }

    fn width(&self) -> f64 {
        let mut w = self.max_x - self.min_x;
        if w < 0.0 {
            w += 360.0;
        }
        w
    }

    fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    fn crosses_date_line(&self) -> bool {
        self.min_x > self.max_x
    }

    fn relate_y_range(&self, min_y: f64, max_y: f64) -> Result<SpatialRelation> {
        Ok(self.relate_y_range_impl(min_y, max_y))
    }

    fn relate_x_range(&self, min_x: f64, max_x: f64) -> Result<SpatialRelation> {
        Ok(self.relate_x_range_impl(min_x, max_x))
    }
}
