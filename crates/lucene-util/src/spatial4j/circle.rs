//! `CircleImpl` and its geodetic subclass `GeoCircle`
//! (`org.locationtech.spatial4j.shape.impl`), one Rust type: a geodetic
//! circle carries [`GeoCircleState`].

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use super::context::SpatialContext;
use super::distance::{DistanceUtils, GeodesicSphereDistCalc, HaversineWithin};
use super::shape::{Circle, Point, Rectangle, Shape, SpatialRelation};
use super::{double_compare_eq, dstr, java_format_fixed, Result};

/// The part of a `GeoCircle` a planar `CircleImpl` does not have.
#[derive(Debug, Clone)]
pub struct GeoCircleState {
    /// `inverseCircle`: the circle around the antipode, when this one
    /// spans more than half the globe (and is not the whole globe).
    inverse_circle: Option<Box<CircleImpl>>,
    /// `horizAxisY`: the latitude of the circle's widest extent.
    horiz_axis_y: f64,
}

/// `CircleImpl` (planar) or `GeoCircle` (geodetic, [`Self::geo`] set).
#[derive(Debug, Clone)]
pub struct CircleImpl {
    point: Arc<dyn Point>,
    radius_deg: f64,
    enclosing_box: Arc<dyn Rectangle>,
    ctx: Arc<SpatialContext>,
    geo: Option<GeoCircleState>,
    /// `contains` precomputed, when the context measures by haversine.
    haversine: Option<HaversineWithin>,
}

/// `Math.ulp(d)`.
pub(crate) fn ulp(d: f64) -> f64 {
    let a = d.abs();
    if a.is_nan() || a.is_infinite() {
        return a;
    }
    if a == f64::MAX {
        return 2f64.powi(971);
    }
    f64::from_bits(a.to_bits() + 1) - a
}

impl CircleImpl {
    /// `new CircleImpl(p, radiusDEG, ctx)`: a planar circle.
    pub fn new(p: Arc<dyn Point>, radius_deg: f64, ctx: Arc<SpatialContext>) -> Result<Self> {
        let radius_deg = if p.is_empty() { f64::NAN } else { radius_deg };
        let enclosing_box = if p.is_empty() {
            ctx.rect(f64::NAN, f64::NAN, f64::NAN, f64::NAN)?
        } else {
            ctx.dist_calc()
                .calc_box_by_dist_from_pt(&p, radius_deg, &ctx)?
        };
        let calc: &dyn Any = &**ctx.dist_calc();
        let haversine = match calc.downcast_ref::<GeodesicSphereDistCalc>() {
            Some(GeodesicSphereDistCalc::Haversine) if !p.is_empty() => {
                Some(HaversineWithin::new(p.x(), p.y(), radius_deg))
            }
            _ => None,
        };
        Ok(CircleImpl {
            point: p,
            radius_deg,
            enclosing_box,
            ctx,
            geo: None,
            haversine,
        })
    }

    /// `new GeoCircle(p, radiusDEG, ctx)`.
    pub fn new_geo(p: Arc<dyn Point>, radius_deg: f64, ctx: Arc<SpatialContext>) -> Result<Self> {
        let mut c = CircleImpl::new(p, radius_deg, ctx)?;
        c.geo_init()?;
        Ok(c)
    }

    /// `GeoCircle.init()`.
    fn geo_init(&mut self) -> Result<()> {
        let state = if self.radius_deg > 90.0 {
            // spans more than half the globe
            let back_dist_deg = 180.0 - self.radius_deg;
            let inverse_circle = if back_dist_deg > 0.0 {
                let mut back_radius = 180.0 - self.radius_deg;
                let back_x = DistanceUtils::norm_lon_deg(self.point.x() + 180.0);
                let back_y = DistanceUtils::norm_lat_deg(self.point.y() + 180.0);
                // Shrink inverseCircle as small as possible to avoid
                // accidental overlap.
                back_radius -= crate::geo::java_max(
                    ulp(back_y.abs() + back_radius),
                    ulp(back_x.abs() + back_radius),
                );
                let p = self.ctx.point_xy(back_x, back_y)?;
                Some(Box::new(CircleImpl::new_geo(
                    p,
                    back_radius,
                    self.ctx.clone(),
                )?))
            } else {
                None // whole globe
            };
            GeoCircleState {
                inverse_circle,
                horiz_axis_y: self.point.y(),
            }
        } else {
            let horiz = self
                .ctx
                .dist_calc()
                .calc_box_by_dist_from_pt_y_horiz_axis_deg(
                    &*self.point,
                    self.radius_deg,
                    &self.ctx,
                )?;
            // some rare numeric conditioning cases can cause this to be
            // barely beyond the box
            let horiz_axis_y = if horiz > self.enclosing_box.max_y() {
                self.enclosing_box.max_y()
            } else if horiz < self.enclosing_box.min_y() {
                self.enclosing_box.min_y()
            } else {
                horiz
            };
            GeoCircleState {
                inverse_circle: None,
                horiz_axis_y,
            }
        };
        self.geo = Some(state);
        Ok(())
    }

    /// Whether this is a `GeoCircle`.
    pub fn is_geo(&self) -> bool {
        self.geo.is_some()
    }

    /// `getCenter()` as the stored point.
    pub fn center_point(&self) -> &Arc<dyn Point> {
        &self.point
    }

    /// `contains(x, y)`.
    pub fn contains(&self, x: f64, y: f64) -> Result<bool> {
        if let Some(h) = &self.haversine {
            return Ok(h.within(x, y));
        }
        self.ctx
            .dist_calc()
            .within(&*self.point, x, y, self.radius_deg)
    }

    /// `getYAxis()`.
    fn y_axis(&self) -> f64 {
        match &self.geo {
            Some(g) => g.horiz_axis_y,
            None => self.point.y(),
        }
    }

    /// `getXAxis()`.
    fn x_axis(&self) -> f64 {
        self.point.x()
    }

    /// `relate(Point)`.
    pub fn relate_point(&self, point: &dyn Point) -> Result<SpatialRelation> {
        Ok(if self.contains(point.x(), point.y())? {
            SpatialRelation::Contains
        } else {
            SpatialRelation::Disjoint
        })
    }

    /// `relate(Rectangle)`.
    pub fn relate_rect(&self, r: &dyn Rectangle) -> Result<SpatialRelation> {
        let bbox_sect = self.enclosing_box.relate(r)?;
        if bbox_sect == SpatialRelation::Disjoint || bbox_sect == SpatialRelation::Within {
            return Ok(bbox_sect);
        } else if bbox_sect == SpatialRelation::Contains && self.enclosing_box.equals(r) {
            // nasty identity edge-case
            return Ok(SpatialRelation::Within);
        }
        self.relate_rectangle_phase2(r, bbox_sect)
    }

    /// `relateRectanglePhase2(r, bboxSect)` (`GeoCircle`'s override first).
    fn relate_rectangle_phase2(
        &self,
        r: &dyn Rectangle,
        bbox_sect: SpatialRelation,
    ) -> Result<SpatialRelation> {
        if let Some(g) = &self.geo {
            if let Some(inverse) = &g.inverse_circle {
                return Ok(inverse.relate(r)?.inverse());
            }
            // if a pole is wrapped, we have a separate algorithm
            if self.enclosing_box.width() == 360.0 {
                return self.relate_rectangle_circle_wraps_pole(r);
            }
            // optimization path for when there are no dateline or pole issues
            if !self.enclosing_box.crosses_date_line() && !r.crosses_date_line() {
                return self.circle_impl_relate_rectangle_phase2(r, bbox_sect);
            }
            // Rectangle wraps around the world longitudinally creating a
            // solid band; there are no corners to test intersection
            if r.width() == 360.0 {
                return Ok(SpatialRelation::Intersects);
            }
            // do quick check to see if all corners are within this circle
            let corners_intersect = self.num_corners_intersect(r)?;
            if corners_intersect == 4 {
                let x_intersect =
                    r.relate_x_range(self.enclosing_box.min_x(), self.enclosing_box.max_x())?;
                if x_intersect == SpatialRelation::Within {
                    return Ok(SpatialRelation::Contains);
                }
                return Ok(SpatialRelation::Intersects);
            }
            if corners_intersect > 0 {
                return Ok(SpatialRelation::Intersects);
            }
            // x axis intersects
            if r.relate_y_range(self.y_axis(), self.y_axis())?.intersects()
                && r.relate_x_range(self.enclosing_box.min_x(), self.enclosing_box.max_x())?
                    .intersects()
            {
                return Ok(SpatialRelation::Intersects);
            }
            // y axis intersects
            if r.relate_x_range(self.x_axis(), self.x_axis())?.intersects() {
                let y_top = self.point.y() + self.radius_deg;
                let y_bot = self.point.y() - self.radius_deg;
                if r.relate_y_range(y_bot, y_top)?.intersects() {
                    return Ok(SpatialRelation::Intersects);
                }
            }
            return Ok(SpatialRelation::Disjoint);
        }
        self.circle_impl_relate_rectangle_phase2(r, bbox_sect)
    }

    /// `CircleImpl.relateRectanglePhase2(r, bboxSect)`.
    fn circle_impl_relate_rectangle_phase2(
        &self,
        r: &dyn Rectangle,
        bbox_sect: SpatialRelation,
    ) -> Result<SpatialRelation> {
        let x_axis = self.x_axis();
        let (closest_x, farthest_x);
        if x_axis < r.min_x() {
            closest_x = r.min_x();
            farthest_x = r.max_x();
        } else if x_axis > r.max_x() {
            closest_x = r.max_x();
            farthest_x = r.min_x();
        } else {
            closest_x = x_axis;
            farthest_x = if r.max_x() - x_axis > x_axis - r.min_x() {
                r.max_x()
            } else {
                r.min_x()
            };
        }
        let y_axis = self.y_axis();
        let (closest_y, farthest_y);
        if y_axis < r.min_y() {
            closest_y = r.min_y();
            farthest_y = r.max_y();
        } else if y_axis > r.max_y() {
            closest_y = r.max_y();
            farthest_y = r.min_y();
        } else {
            closest_y = y_axis;
            farthest_y = if r.max_y() - y_axis > y_axis - r.min_y() {
                r.max_y()
            } else {
                r.min_y()
            };
        }
        // If r doesn't overlap an axis, then could be disjoint.
        if x_axis != closest_x && y_axis != closest_y && !self.contains(closest_x, closest_y)? {
            return Ok(SpatialRelation::Disjoint);
        }
        if bbox_sect != SpatialRelation::Contains {
            return Ok(SpatialRelation::Intersects);
        }
        if !self.contains(farthest_x, farthest_y)? {
            return Ok(SpatialRelation::Intersects);
        }
        // geodetic detection of farthest Y when rect crosses x axis can't be
        // reliably determined, so check other corner too
        if self.point.y() != self.y_axis() && y_axis == closest_y {
            let other_y = if farthest_y == r.max_y() {
                r.min_y()
            } else {
                r.max_y()
            };
            if !self.contains(farthest_x, other_y)? {
                return Ok(SpatialRelation::Intersects);
            }
        }
        Ok(SpatialRelation::Contains)
    }

    /// `GeoCircle.relateRectangleCircleWrapsPole(r, ctx)`.
    fn relate_rectangle_circle_wraps_pole(&self, r: &dyn Rectangle) -> Result<SpatialRelation> {
        if self.radius_deg == 180.0 {
            return Ok(SpatialRelation::Contains);
        }
        // Check if r is within the pole wrap region:
        let y_top = self.point.y() + self.radius_deg;
        if y_top > 90.0 {
            let y_top_overlap = y_top - 90.0;
            if r.min_y() >= 90.0 - y_top_overlap {
                return Ok(SpatialRelation::Contains);
            }
        } else {
            let y_bot = self.point.y() - self.radius_deg;
            if y_bot < -90.0 {
                let y_bot_overlap = -90.0 - y_bot;
                if r.max_y() <= -90.0 + y_bot_overlap {
                    return Ok(SpatialRelation::Contains);
                }
            }
        }
        if r.width() == 360.0 {
            return Ok(SpatialRelation::Intersects);
        }
        let corners_intersect = self.num_corners_intersect(r)?;
        let front_x = self.point.x();
        if corners_intersect == 4 {
            let back_x = if front_x <= 0.0 {
                front_x + 180.0
            } else {
                front_x - 180.0
            };
            if r.relate_x_range(back_x, back_x)?.intersects() {
                Ok(SpatialRelation::Intersects)
            } else {
                Ok(SpatialRelation::Contains)
            }
        } else if corners_intersect == 0 {
            if r.relate_x_range(front_x, front_x)?.intersects() {
                Ok(SpatialRelation::Intersects)
            } else {
                Ok(SpatialRelation::Disjoint)
            }
        } else {
            Ok(SpatialRelation::Intersects)
        }
    }

    /// `GeoCircle.numCornersIntersect(r)`: 0 for none, 1 for some, 4 for
    /// all.
    fn num_corners_intersect(&self, r: &dyn Rectangle) -> Result<u8> {
        let all = self.contains(r.min_x(), r.min_y())?;
        for (x, y) in [
            (r.min_x(), r.max_y()),
            (r.max_x(), r.min_y()),
            (r.max_x(), r.max_y()),
        ] {
            if self.contains(x, y)? != all {
                return Ok(1);
            }
        }
        Ok(if all { 4 } else { 0 })
    }

    /// `relate(Circle)`.
    pub fn relate_circle(&self, circle: &dyn Circle) -> Result<SpatialRelation> {
        let cross_dist = self
            .ctx
            .dist_calc()
            .distance(&*self.point, &*circle.center()?)?;
        let a_dist = self.radius_deg;
        let b_dist = circle.radius();
        if cross_dist > a_dist + b_dist {
            return Ok(SpatialRelation::Disjoint);
        }
        if cross_dist < a_dist && cross_dist + b_dist <= a_dist {
            return Ok(SpatialRelation::Contains);
        }
        if cross_dist < b_dist && cross_dist + a_dist <= b_dist {
            return Ok(SpatialRelation::Within);
        }
        Ok(SpatialRelation::Intersects)
    }
}

impl fmt::Display for CircleImpl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.geo.is_some() {
            let dist_km =
                DistanceUtils::degrees2_dist(self.radius_deg, DistanceUtils::EARTH_MEAN_RADIUS_KM);
            write!(
                f,
                "Circle({}, d={}\u{00B0} {}km)",
                self.point,
                java_format_fixed(self.radius_deg, 1),
                java_format_fixed(dist_km, 2)
            )
        } else {
            write!(
                f,
                "Circle({}, d={}\u{00B0})",
                self.point,
                dstr(self.radius_deg)
            )
        }
    }
}

impl Shape for CircleImpl {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        if self.is_empty() || other.is_empty() {
            return Ok(SpatialRelation::Disjoint);
        }
        if let Some(p) = other.as_point() {
            return self.relate_point(p);
        }
        if let Some(r) = other.as_rectangle() {
            return self.relate_rect(r);
        }
        if let Some(c) = other.as_circle() {
            return self.relate_circle(c);
        }
        Ok(other.relate(self)?.transpose())
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        Ok(self.enclosing_box.clone())
    }

    fn has_area(&self) -> bool {
        self.radius_deg > 0.0
    }

    fn area(&self, ctx: Option<&SpatialContext>) -> Result<f64> {
        match ctx {
            None => Ok(std::f64::consts::PI * self.radius_deg * self.radius_deg),
            Some(ctx) => ctx.dist_calc().area_circle(self),
        }
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        Ok(self.point.clone())
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        Ok(ctx.circle_at(&self.point, distance + self.radius_deg)?)
    }

    fn is_empty(&self) -> bool {
        self.point.is_empty()
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        circle_equals(self, other)
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_circle(&self) -> Option<&dyn Circle> {
        Some(self)
    }
}

impl Circle for CircleImpl {
    fn radius(&self) -> f64 {
        self.radius_deg
    }
}

/// `CircleImpl.equals(Circle, Object)`: any circle with an equal center
/// and the same radius.
pub fn circle_equals(thiz: &dyn Circle, o: &dyn Shape) -> bool {
    let Some(c) = o.as_circle() else {
        return false;
    };
    let (Ok(a), Ok(b)) = (thiz.center(), c.center()) else {
        return false;
    };
    a.equals(&*b) && double_compare_eq(c.radius(), thiz.radius())
}
