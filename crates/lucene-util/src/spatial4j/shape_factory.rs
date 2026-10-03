//! `ShapeFactory` and its builders (`org.locationtech.spatial4j.shape`),
//! and the default `ShapeFactoryImpl`
//! (`org.locationtech.spatial4j.shape.impl`), which builds no polygons.

use std::fmt;
use std::sync::Arc;

use super::buffered_line::BufferedLineString;
use super::circle::CircleImpl;
use super::collection::ShapeCollection;
use super::context::SpatialContext;
use super::distance::DistanceUtils;
use super::point::PointImpl;
use super::rectangle::RectangleImpl;
use super::shape::{Circle, Point, Rectangle, Shape, SpatialRelation};
use super::{dstr, Error, Result};

/// `ShapeFactory`. Every method takes the context the shapes are made in
/// (Java's factory holds it).
pub trait ShapeFactory: Send + Sync + fmt::Debug {
    /// `isNormWrapLongitude()`.
    fn is_norm_wrap_longitude(&self) -> bool;
    /// `normX(x)`.
    fn norm_x(&self, x: f64) -> f64;
    /// `normY(y)`.
    fn norm_y(&self, y: f64) -> f64 {
        y
    }
    /// `normZ(z)`.
    fn norm_z(&self, z: f64) -> f64 {
        z
    }
    /// `normDist(d)`.
    fn norm_dist(&self, d: f64) -> f64 {
        d
    }
    /// `verifyX(x)`.
    fn verify_x(&self, ctx: &Arc<SpatialContext>, x: f64) -> Result<()> {
        let [min_x, max_x, _, _] = ctx.world_bounds_values();
        if x < min_x || x > max_x {
            return Err(Error::InvalidShape(format!(
                "Bad X value {} is not in boundary {}",
                dstr(x),
                ctx.world_bounds()
            )));
        }
        Ok(())
    }
    /// `verifyY(y)`.
    fn verify_y(&self, ctx: &Arc<SpatialContext>, y: f64) -> Result<()> {
        let [_, _, min_y, max_y] = ctx.world_bounds_values();
        if y < min_y || y > max_y {
            return Err(Error::InvalidShape(format!(
                "Bad Y value {} is not in boundary {}",
                dstr(y),
                ctx.world_bounds()
            )));
        }
        Ok(())
    }
    /// `pointXY(x, y)`.
    fn point_xy(&self, ctx: &Arc<SpatialContext>, x: f64, y: f64) -> Result<Arc<dyn Point>>;
    /// A point made without verification: what Java's `Point.reset(x, y)`
    /// leaves behind.
    fn point_xy_unchecked(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
    ) -> Result<Arc<dyn Point>>;
    /// `pointXYZ(x, y, z)`.
    fn point_xyz(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<Arc<dyn Point>>;
    /// `pointLatLon(latitude, longitude)`.
    fn point_lat_lon(
        &self,
        ctx: &Arc<SpatialContext>,
        latitude: f64,
        longitude: f64,
    ) -> Result<Arc<dyn Point>> {
        self.point_xy(ctx, longitude, latitude)
    }
    /// `rect(minX, maxX, minY, maxY)`.
    fn rect(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Result<Arc<dyn Rectangle>>;
    /// `rect(minX, maxX, minY, maxY).relate(other)`: the rectangle need not
    /// outlive the call, so a factory may skip allocating it.
    fn rect_relate(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        other: &dyn Shape,
    ) -> Result<SpatialRelation> {
        self.rect(ctx, min_x, max_x, min_y, max_y)?.relate(other)
    }
    /// `circle(x, y, distance)`.
    fn circle(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        distance: f64,
    ) -> Result<Arc<dyn Circle>>;
    /// `circle(point, distance)`.
    fn circle_at(
        &self,
        ctx: &Arc<SpatialContext>,
        point: &Arc<dyn Point>,
        distance: f64,
    ) -> Result<Arc<dyn Circle>>;
    /// `lineString(points, buf)`.
    fn line_string(
        &self,
        ctx: &Arc<SpatialContext>,
        points: &[Arc<dyn Point>],
        buf: f64,
    ) -> Result<Arc<dyn Shape>>;
    /// `multiShape(coll)`.
    fn multi_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        coll: Vec<Arc<dyn Shape>>,
    ) -> Result<ShapeCollection>;
    /// `lineString()` (the builder).
    fn line_string_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn LineStringBuilder>;
    /// `polygon()` (the builder).
    fn polygon_builder(&self, ctx: &Arc<SpatialContext>) -> Result<Box<dyn PolygonBuilder>>;
    /// `multiShape(Shape.class)` (the builder).
    fn multi_shape_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiShapeBuilder>;
    /// `multiPoint()`.
    fn multi_point_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPointBuilder>;
    /// `multiLineString()`.
    fn multi_line_string_builder(
        &self,
        ctx: &Arc<SpatialContext>,
    ) -> Box<dyn MultiLineStringBuilder>;
    /// `multiPolygon()`.
    fn multi_polygon_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPolygonBuilder>;
    /// `this instanceof S2ShapeFactory`: the cell shapes `S2PrefixTree`
    /// needs.
    fn as_s2(&self) -> Option<&dyn crate::spatial_extras::prefix_tree::S2ShapeFactory> {
        None
    }
}

/// `ShapeFactory.PointsBuilder`.
pub trait PointsBuilder {
    /// `pointXY(x, y)`.
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()>;
    /// `pointXYZ(x, y, z)`.
    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()>;
    /// `pointLatLon(latitude, longitude)`.
    fn point_lat_lon(&mut self, latitude: f64, longitude: f64) -> Result<()> {
        self.point_xy(longitude, latitude)
    }
}

/// `ShapeFactory.LineStringBuilder`.
pub trait LineStringBuilder: PointsBuilder {
    /// `buffer(distance)`.
    fn buffer(&mut self, distance: f64);
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
    /// Up-cast for the WKT reader's generic point list.
    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder;
}

/// `ShapeFactory.PolygonBuilder.HoleBuilder`.
pub trait HoleBuilder: PointsBuilder {
    /// `endHole()`.
    fn end_hole(self: Box<Self>) -> Result<()>;
    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder;
}

/// `ShapeFactory.PolygonBuilder`.
pub trait PolygonBuilder: PointsBuilder {
    /// `hole()`: the hole builder; [`HoleBuilder::end_hole`] adds it to this
    /// polygon.
    fn hole(&mut self) -> Box<dyn HoleBuilder + '_>;
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
    /// `buildOrRect()`.
    fn build_or_rect(self: Box<Self>) -> Result<Arc<dyn Shape>>;
    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder;
}

/// `ShapeFactory.MultiShapeBuilder`.
pub trait MultiShapeBuilder {
    /// `add(shape)`.
    fn add(&mut self, shape: Arc<dyn Shape>) -> Result<()>;
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
}

/// `ShapeFactory.MultiPointBuilder`.
pub trait MultiPointBuilder: PointsBuilder {
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder;
}

/// `ShapeFactory.MultiLineStringBuilder`.
pub trait MultiLineStringBuilder {
    /// `lineString()`.
    fn line_string(&self) -> Box<dyn LineStringBuilder>;
    /// `add(lineStringBuilder)`.
    fn add(&mut self, builder: Box<dyn LineStringBuilder>) -> Result<()>;
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
}

/// `ShapeFactory.MultiPolygonBuilder`.
pub trait MultiPolygonBuilder {
    /// `polygon()`.
    fn polygon(&self) -> Result<Box<dyn PolygonBuilder>>;
    /// `add(polygonBuilder)`.
    fn add(&mut self, builder: Box<dyn PolygonBuilder>) -> Result<()>;
    /// `build()`.
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>>;
}

/// `ShapeFactoryImpl`: points, rectangles, circles (`GeoCircle` for geo),
/// buffered line strings and shape collections; no polygons.
#[derive(Debug, Clone)]
pub struct ShapeFactoryImpl {
    norm_wrap_longitude: bool,
}

impl ShapeFactoryImpl {
    /// `new ShapeFactoryImpl(ctx, factory)`: `normWrapLongitude` is
    /// `ctx.isGeo() && factory.normWrapLongitude`.
    pub fn new(norm_wrap_longitude: bool) -> Self {
        ShapeFactoryImpl {
            norm_wrap_longitude,
        }
    }

    /// `rect(minX, maxX, minY, maxY)`'s validation and dateline handling,
    /// the rectangle unboxed.
    fn rect_impl(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Result<RectangleImpl> {
        let [bminx, bmaxx, bminy, bmaxy] = ctx.world_bounds_values();
        if min_y < bminy || max_y > bmaxy {
            return Err(Error::InvalidShape(format!(
                "Y values [{} to {}] not in boundary {}",
                dstr(min_y),
                dstr(max_y),
                ctx.world_bounds()
            )));
        }
        if min_y > max_y {
            return Err(Error::InvalidShape(format!(
                "maxY must be >= minY: {} to {}",
                dstr(min_y),
                dstr(max_y)
            )));
        }
        let (mut min_x, mut max_x) = (min_x, max_x);
        if ctx.is_geo() {
            self.verify_x(ctx, min_x)?;
            self.verify_x(ctx, max_x)?;
            // If an edge coincides with the dateline then don't make this
            // rect cross it.
            if min_x == 180.0 && min_x != max_x {
                min_x = -180.0;
            } else if max_x == -180.0 && min_x != max_x {
                max_x = 180.0;
            }
        } else {
            if min_x < bminx || max_x > bmaxx {
                return Err(Error::InvalidShape(format!(
                    "X values [{} to {}] not in boundary {}",
                    dstr(min_x),
                    dstr(max_x),
                    ctx.world_bounds()
                )));
            }
            if min_x > max_x {
                return Err(Error::InvalidShape(format!(
                    "maxX must be >= minX: {} to {}",
                    dstr(min_x),
                    dstr(max_x)
                )));
            }
        }
        Ok(RectangleImpl::new(min_x, max_x, min_y, max_y, ctx.clone()))
    }
}

impl ShapeFactory for ShapeFactoryImpl {
    fn is_norm_wrap_longitude(&self) -> bool {
        self.norm_wrap_longitude
    }

    fn norm_x(&self, x: f64) -> f64 {
        if self.norm_wrap_longitude {
            DistanceUtils::norm_lon_deg(x)
        } else {
            x
        }
    }

    fn point_xy(&self, ctx: &Arc<SpatialContext>, x: f64, y: f64) -> Result<Arc<dyn Point>> {
        self.verify_x(ctx, x)?;
        self.verify_y(ctx, y)?;
        Ok(Arc::new(PointImpl::new(x, y, ctx.clone())))
    }

    fn point_xy_unchecked(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
    ) -> Result<Arc<dyn Point>> {
        Ok(Arc::new(PointImpl::new(x, y, ctx.clone())))
    }

    fn point_xyz(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        _z: f64,
    ) -> Result<Arc<dyn Point>> {
        self.point_xy(ctx, x, y)
    }

    fn rect(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Result<Arc<dyn Rectangle>> {
        Ok(Arc::new(self.rect_impl(ctx, min_x, max_x, min_y, max_y)?))
    }

    fn rect_relate(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        other: &dyn Shape,
    ) -> Result<SpatialRelation> {
        self.rect_impl(ctx, min_x, max_x, min_y, max_y)?
            .relate(other)
    }

    fn circle(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        distance: f64,
    ) -> Result<Arc<dyn Circle>> {
        let p = self.point_xy(ctx, x, y)?;
        self.circle_at(ctx, &p, distance)
    }

    fn circle_at(
        &self,
        ctx: &Arc<SpatialContext>,
        point: &Arc<dyn Point>,
        distance: f64,
    ) -> Result<Arc<dyn Circle>> {
        if distance < 0.0 {
            return Err(Error::InvalidShape(format!(
                "distance must be >= 0; got {}",
                dstr(distance)
            )));
        }
        if ctx.is_geo() {
            let distance = if distance > 180.0 { 180.0 } else { distance };
            Ok(Arc::new(CircleImpl::new_geo(
                point.clone(),
                distance,
                ctx.clone(),
            )?))
        } else {
            Ok(Arc::new(CircleImpl::new(
                point.clone(),
                distance,
                ctx.clone(),
            )?))
        }
    }

    fn line_string(
        &self,
        ctx: &Arc<SpatialContext>,
        points: &[Arc<dyn Point>],
        buf: f64,
    ) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(BufferedLineString::new(
            points,
            buf,
            ctx.is_geo(),
            ctx.clone(),
        )?))
    }

    fn multi_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        coll: Vec<Arc<dyn Shape>>,
    ) -> Result<ShapeCollection> {
        ShapeCollection::new(coll, ctx.clone())
    }

    fn line_string_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn LineStringBuilder> {
        Box::new(ImplLineStringBuilder {
            ctx: ctx.clone(),
            points: Vec::new(),
            buffer_distance: 0.0,
        })
    }

    fn polygon_builder(&self, _ctx: &Arc<SpatialContext>) -> Result<Box<dyn PolygonBuilder>> {
        Err(Error::UnsupportedOperation(Some(
            "Unsupported shape of this SpatialContext. Try JTS or Geo3D.".into(),
        )))
    }

    fn multi_shape_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiShapeBuilder> {
        Box::new(GeneralShapeMultiShapeBuilder::new(ctx))
    }

    fn multi_point_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPointBuilder> {
        Box::new(GeneralShapeMultiShapeBuilder::new(ctx))
    }

    fn multi_line_string_builder(
        &self,
        ctx: &Arc<SpatialContext>,
    ) -> Box<dyn MultiLineStringBuilder> {
        Box::new(GeneralShapeMultiShapeBuilder::new(ctx))
    }

    fn multi_polygon_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPolygonBuilder> {
        Box::new(GeneralShapeMultiShapeBuilder::new(ctx))
    }
}

/// `ShapeFactoryImpl.lineString()`'s builder.
struct ImplLineStringBuilder {
    ctx: Arc<SpatialContext>,
    points: Vec<Arc<dyn Point>>,
    buffer_distance: f64,
}

impl PointsBuilder for ImplLineStringBuilder {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.points.push(self.ctx.point_xy(x, y)?);
        Ok(())
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.points
            .push(self.ctx.shape_factory().point_xyz(&self.ctx, x, y, z)?);
        Ok(())
    }
}

impl LineStringBuilder for ImplLineStringBuilder {
    fn buffer(&mut self, distance: f64) {
        self.buffer_distance = distance;
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(BufferedLineString::new(
            &self.points,
            self.buffer_distance,
            false,
            self.ctx.clone(),
        )?))
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

/// `ShapeFactoryImpl.GeneralShapeMultiShapeBuilder`: every multi-shape
/// builder, building a [`ShapeCollection`].
struct GeneralShapeMultiShapeBuilder {
    ctx: Arc<SpatialContext>,
    shapes: Vec<Arc<dyn Shape>>,
}

impl GeneralShapeMultiShapeBuilder {
    fn new(ctx: &Arc<SpatialContext>) -> Self {
        GeneralShapeMultiShapeBuilder {
            ctx: ctx.clone(),
            shapes: Vec::new(),
        }
    }

    fn build_collection(self) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(ShapeCollection::new(self.shapes, self.ctx)?))
    }
}

impl PointsBuilder for GeneralShapeMultiShapeBuilder {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.shapes.push(self.ctx.point_xy(x, y)?);
        Ok(())
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.shapes
            .push(self.ctx.shape_factory().point_xyz(&self.ctx, x, y, z)?);
        Ok(())
    }
}

impl MultiShapeBuilder for GeneralShapeMultiShapeBuilder {
    fn add(&mut self, shape: Arc<dyn Shape>) -> Result<()> {
        self.shapes.push(shape);
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        (*self).build_collection()
    }
}

impl MultiPointBuilder for GeneralShapeMultiShapeBuilder {
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        (*self).build_collection()
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

impl MultiLineStringBuilder for GeneralShapeMultiShapeBuilder {
    fn line_string(&self) -> Box<dyn LineStringBuilder> {
        self.ctx.shape_factory().line_string_builder(&self.ctx)
    }

    fn add(&mut self, builder: Box<dyn LineStringBuilder>) -> Result<()> {
        self.shapes.push(builder.build()?);
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        (*self).build_collection()
    }
}

impl MultiPolygonBuilder for GeneralShapeMultiShapeBuilder {
    fn polygon(&self) -> Result<Box<dyn PolygonBuilder>> {
        self.ctx.shape_factory().polygon_builder(&self.ctx)
    }

    fn add(&mut self, builder: Box<dyn PolygonBuilder>) -> Result<()> {
        self.shapes.push(builder.build()?);
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        (*self).build_collection()
    }
}
