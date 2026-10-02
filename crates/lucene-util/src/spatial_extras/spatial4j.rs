//! Lucene's bridge from Spatial4j to spatial3d
//! (`org.apache.lucene.spatial.spatial4j`): `Geo3dShape` and its point,
//! rectangle and circle specialisations, `Geo3dShapeFactory` (which can
//! build polygons, paths and S2 cells), `Geo3dBinaryCodec`,
//! `Geo3dDistanceCalculator`. `Geo3dSpatialContextFactory` is
//! [`SpatialContextFactory::geo3d`](crate::spatial4j::SpatialContextFactory::geo3d).
//!
//! # Deviations
//!
//! - `Geo3dShape.equals` compares the wrapped shapes' serialized forms
//!   (class and fields), which is what geo3d's `equals` compares; geo3d
//!   shapes are not `PartialEq` in this port.
//! - `toString` is `Geo3D:` and the geo3d class name (geo3d's own
//!   `toString`s are not ported).
//!
//! # A Lucene quirk kept
//!
//! `Geo3dShape.relate` reads `GeoArea.getRelationship`'s answer (how the
//! *argument* shape relates to the area) the same way whichever side the
//! Geo3D shape is on. For another Geo3D shape the area is `this`, which is
//! right; for a plain Spatial4j rectangle the area is the rectangle, so a
//! Geo3D shape containing a `RectangleImpl` comes out `WITHIN` and one
//! within it `CONTAINS`. This port answers as Lucene does (the world bounds
//! a Geo3D context hands out are such a rectangle).

use std::any::Any;
use std::fmt;
use std::sync::{Arc, OnceLock};

use crate::s2::{S2Cell, S2CellId, S2Point};
use crate::spatial3d::geo_area_factory::make_geo_area_lat_lon;
use crate::spatial3d::geo_bbox_factory::{make_geo_bbox, make_geo_bbox_from_bounds};
use crate::spatial3d::geo_circle_factory::{make_exact_geo_circle, make_geo_circle};
use crate::spatial3d::geo_composite::GeoCompositeAreaShape;
use crate::spatial3d::geo_path_factory::make_geo_path;
use crate::spatial3d::geo_polygon_factory::{
    make_geo_polygon_from_description, PolygonDescription,
};
use crate::spatial3d::geo_s2_shape::{make_geo_point_shape, make_geo_s2_shape};
use crate::spatial3d::standard_objects::{read_object, write_object, StandardObject, CLASS_NAMES};
use crate::spatial3d::{
    GeoAreaRelationship, GeoAreaShape, GeoBBox, GeoCircle, GeoPoint, GeoPointShape, LatLonBounds,
    PlanetModel,
};
use crate::spatial4j::binary_codec::{BinaryCodec, DataInput};
use crate::spatial4j::shape_factory::{
    HoleBuilder, LineStringBuilder, MultiLineStringBuilder, MultiPointBuilder, MultiPolygonBuilder,
    MultiShapeBuilder, PointsBuilder, PolygonBuilder, ShapeFactory,
};
use crate::spatial4j::{
    Circle, DistanceCalculator, DistanceUtils, Error, Point, Rectangle, Result, Shape,
    ShapeCollection, SpatialContext, SpatialRelation,
};

use super::prefix_tree::S2ShapeFactory;

const D2R: f64 = DistanceUtils::DEGREES_TO_RADIANS;
const R2D: f64 = DistanceUtils::RADIANS_TO_DEGREES;

/// What `instanceof` a Geo3D shape answers to besides `Geo3dShape`.
#[derive(Clone)]
enum Kind {
    /// A plain `Geo3dShape`.
    Plain,
    /// `Geo3dPointShape`.
    Point(Arc<dyn GeoPointShape>),
    /// `Geo3dRectangleShape`, with its degree bounds.
    Rectangle {
        bbox: Arc<dyn GeoBBox>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    },
    /// `Geo3dCircleShape`.
    Circle(Arc<dyn GeoCircle>),
}

/// `Geo3dShape` (and `Geo3dPointShape`, `Geo3dRectangleShape`,
/// `Geo3dCircleShape`): a Spatial4j shape wrapping a geo3d area shape.
pub struct Geo3dShape {
    ctx: Arc<SpatialContext>,
    shape: Arc<dyn GeoAreaShape>,
    kind: Kind,
    bounding_box: OnceLock<Arc<dyn Rectangle>>,
    center: OnceLock<Arc<dyn Point>>,
}

impl fmt::Debug for Geo3dShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

impl Clone for Geo3dShape {
    fn clone(&self) -> Self {
        Geo3dShape {
            ctx: self.ctx.clone(),
            shape: self.shape.clone(),
            kind: self.kind.clone(),
            bounding_box: self.bounding_box.clone(),
            center: self.center.clone(),
        }
    }
}

/// The class name `toString` and the binary codec's messages use.
fn geo3d_simple_name(shape: &dyn GeoAreaShape) -> &'static str {
    shape
        .class_code()
        .and_then(|c| CLASS_NAMES.get(c as usize).copied())
        .unwrap_or("GeoAreaShape")
}

impl Geo3dShape {
    /// `new Geo3dShape(shape, ctx)`.
    pub fn new(shape: Arc<dyn GeoAreaShape>, ctx: Arc<SpatialContext>) -> Self {
        Geo3dShape {
            ctx,
            shape,
            kind: Kind::Plain,
            bounding_box: OnceLock::new(),
            center: OnceLock::new(),
        }
    }

    /// `new Geo3dPointShape(shape, ctx)`.
    pub fn new_point(shape: Arc<dyn GeoPointShape>, ctx: Arc<SpatialContext>) -> Self {
        Geo3dShape {
            ctx,
            shape: shape.clone(),
            kind: Kind::Point(shape),
            bounding_box: OnceLock::new(),
            center: OnceLock::new(),
        }
    }

    /// `new Geo3dRectangleShape(shape, ctx, minX, maxX, minY, maxY)`.
    pub fn new_rectangle_with_bounds(
        shape: Arc<dyn GeoBBox>,
        ctx: Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Self {
        Geo3dShape {
            ctx,
            shape: shape.clone(),
            kind: Kind::Rectangle {
                bbox: shape,
                min_x,
                max_x,
                min_y,
                max_y,
            },
            bounding_box: OnceLock::new(),
            center: OnceLock::new(),
        }
    }

    /// `new Geo3dRectangleShape(shape, ctx)`: the bounds from the box's own
    /// (`setBoundsFromshape`).
    pub fn new_rectangle(shape: Arc<dyn GeoBBox>, ctx: Arc<SpatialContext>) -> Result<Self> {
        let mut bounds = LatLonBounds::new();
        shape.get_bounds(&mut bounds);
        let npe = || Error::NullPointer("null".into());
        let min_x = if bounds.check_no_longitude_bound() {
            -180.0
        } else {
            bounds.left_longitude().ok_or_else(npe)? * R2D
        };
        let min_y = if bounds.check_no_bottom_latitude_bound() {
            -90.0
        } else {
            bounds.min_latitude().ok_or_else(npe)? * R2D
        };
        let max_x = if bounds.check_no_longitude_bound() {
            180.0
        } else {
            bounds.right_longitude().ok_or_else(npe)? * R2D
        };
        let max_y = if bounds.check_no_top_latitude_bound() {
            90.0
        } else {
            bounds.max_latitude().ok_or_else(npe)? * R2D
        };
        Ok(Self::new_rectangle_with_bounds(
            shape, ctx, min_x, max_x, min_y, max_y,
        ))
    }

    /// `new Geo3dCircleShape(shape, ctx)`.
    pub fn new_circle(shape: Arc<dyn GeoCircle>, ctx: Arc<SpatialContext>) -> Self {
        Geo3dShape {
            ctx,
            shape: shape.clone(),
            kind: Kind::Circle(shape),
            bounding_box: OnceLock::new(),
            center: OnceLock::new(),
        }
    }

    /// The wrapped geo3d shape (Java's `shape` field).
    pub fn geo_shape(&self) -> &Arc<dyn GeoAreaShape> {
        &self.shape
    }

    /// The wrapped point, for a `Geo3dPointShape`.
    pub fn geo_point_shape(&self) -> Option<&Arc<dyn GeoPointShape>> {
        match &self.kind {
            Kind::Point(p) => Some(p),
            _ => None,
        }
    }

    fn pm(&self) -> &Arc<PlanetModel> {
        self.shape.planet_model()
    }

    /// The Java class this stands for.
    pub fn java_class(&self) -> &'static str {
        match self.kind {
            Kind::Plain => "org.apache.lucene.spatial.spatial4j.Geo3dShape",
            Kind::Point(_) => "org.apache.lucene.spatial.spatial4j.Geo3dPointShape",
            Kind::Rectangle { .. } => "org.apache.lucene.spatial.spatial4j.Geo3dRectangleShape",
            Kind::Circle(_) => "org.apache.lucene.spatial.spatial4j.Geo3dCircleShape",
        }
    }

    /// `relate(Rectangle)`: the box as a geo3d area, related to the shape.
    fn relate_rect(&self, r: &dyn Rectangle) -> Result<GeoAreaRelationship> {
        let area = make_geo_area_lat_lon(
            self.pm(),
            r.max_y() * D2R,
            r.min_y() * D2R,
            r.min_x() * D2R,
            r.max_x() * D2R,
        )?;
        Ok(area.get_relationship(&*self.shape)?)
    }

    /// `relate(Point)`.
    fn relate_point(&self, p: &dyn Point) -> Result<GeoAreaRelationship> {
        let point = GeoPoint::from_lat_lon(self.pm(), p.y() * D2R, p.x() * D2R)?;
        Ok(if self.shape.is_within(&point) {
            GeoAreaRelationship::Within
        } else {
            GeoAreaRelationship::Disjoint
        })
    }
}

/// The geo3d shape inside a Spatial4j shape, when it is a [`Geo3dShape`]
/// (`other instanceof Geo3dShape`).
pub fn geo3d_area(s: &dyn Shape) -> Option<&Arc<dyn GeoAreaShape>> {
    s.as_any().downcast_ref::<Geo3dShape>().map(|g| &g.shape)
}

/// The Java class name of a Geo3D shape, for messages.
pub fn geo3d_class_name(s: &dyn Shape) -> Option<&'static str> {
    s.as_any()
        .downcast_ref::<Geo3dShape>()
        .map(|g| g.java_class())
}

fn serialized(shape: &dyn GeoAreaShape) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    write_object(&mut out, shape).ok()?;
    Some(out)
}

impl fmt::Display for Geo3dShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Geo3D:{}", geo3d_simple_name(&*self.shape))
    }
}

impl Shape for Geo3dShape {
    fn relate(&self, other: &dyn Shape) -> Result<SpatialRelation> {
        let relationship = if let Some(s) = geo3d_area(other) {
            self.shape.get_relationship(&**s)?
        } else if let Some(r) = other.as_rectangle() {
            self.relate_rect(r)?
        } else if let Some(p) = other.as_point() {
            self.relate_point(p)?
        } else {
            return Err(Error::Runtime(format!(
                "Unimplemented shape relationship determination: class {}",
                crate::spatial4j::binary_codec::java_class_name(other)
            )));
        };
        let is_point = other.as_point().is_some();
        Ok(match relationship {
            GeoAreaRelationship::Disjoint => SpatialRelation::Disjoint,
            GeoAreaRelationship::Overlaps => {
                if is_point {
                    SpatialRelation::Contains
                } else {
                    SpatialRelation::Intersects
                }
            }
            GeoAreaRelationship::Contains => {
                if is_point {
                    SpatialRelation::Contains
                } else {
                    SpatialRelation::Within
                }
            }
            GeoAreaRelationship::Within => SpatialRelation::Contains,
        })
    }

    fn bounding_box(&self) -> Result<Arc<dyn Rectangle>> {
        if let Some(b) = self.bounding_box.get() {
            return Ok(b.clone());
        }
        let bbox: Arc<dyn Rectangle> = match &self.kind {
            Kind::Rectangle { .. } => Arc::new(self.clone()),
            Kind::Point(p) => Arc::new(Geo3dShape::new_rectangle(
                p.clone() as Arc<dyn GeoBBox>,
                self.ctx.clone(),
            )?),
            _ => {
                let mut bounds = LatLonBounds::new();
                self.shape.get_bounds(&mut bounds);
                let geo_bbox = make_geo_bbox_from_bounds(self.pm(), &bounds)?;
                Arc::new(Geo3dShape::new_rectangle(geo_bbox, self.ctx.clone())?)
            }
        };
        Ok(self.bounding_box.get_or_init(|| bbox).clone())
    }

    fn has_area(&self) -> bool {
        !matches!(self.kind, Kind::Point(_))
    }

    fn area(&self, _ctx: Option<&SpatialContext>) -> Result<f64> {
        Err(Error::UnsupportedOperation(None))
    }

    fn center(&self) -> Result<Arc<dyn Point>> {
        if let Some(c) = self.center.get() {
            return Ok(c.clone());
        }
        let center: Arc<dyn Point> = match &self.kind {
            Kind::Point(_) => Arc::new(self.clone()),
            Kind::Rectangle { bbox, .. } => {
                let point = bbox.center();
                Arc::new(Geo3dShape::new_point(
                    make_geo_point_shape(self.pm(), point.latitude(), point.longitude())?,
                    self.ctx.clone(),
                ))
            }
            Kind::Circle(c) => {
                let point = c.center();
                Arc::new(Geo3dShape::new_point(
                    make_geo_point_shape(self.pm(), point.latitude(), point.longitude())?,
                    self.ctx.clone(),
                ))
            }
            Kind::Plain => self.bounding_box()?.center()?,
        };
        Ok(self.center.get_or_init(|| center).clone())
    }

    fn buffered(&self, distance: f64, ctx: &Arc<SpatialContext>) -> Result<Arc<dyn Shape>> {
        match &self.kind {
            Kind::Point(_) => {
                let p = self.as_point().expect("a point");
                Ok(ctx.circle(p.x(), p.y(), distance)?)
            }
            Kind::Rectangle { bbox, .. } => {
                let expanded = bbox.expand(distance * D2R)?;
                Ok(Arc::new(Geo3dShape::new_rectangle(expanded, ctx.clone())?))
            }
            _ => Err(Error::UnsupportedOperation(None)),
        }
    }

    fn is_empty(&self) -> bool {
        false
    }

    fn equals(&self, other: &dyn Shape) -> bool {
        let Some(o) = other.as_any().downcast_ref::<Geo3dShape>() else {
            return false;
        };
        Arc::ptr_eq(&o.ctx, &self.ctx)
            && serialized(&*o.shape).is_some_and(|b| Some(b) == serialized(&*self.shape))
    }

    fn context(&self) -> Option<&Arc<SpatialContext>> {
        Some(&self.ctx)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_point(&self) -> Option<&dyn Point> {
        matches!(self.kind, Kind::Point(_)).then_some(self as &dyn Point)
    }

    fn as_rectangle(&self) -> Option<&dyn Rectangle> {
        matches!(self.kind, Kind::Rectangle { .. }).then_some(self as &dyn Rectangle)
    }

    fn as_circle(&self) -> Option<&dyn Circle> {
        matches!(self.kind, Kind::Circle(_)).then_some(self as &dyn Circle)
    }
}

impl Point for Geo3dShape {
    fn x(&self) -> f64 {
        match &self.kind {
            Kind::Point(p) => p.center().longitude() * R2D,
            _ => f64::NAN,
        }
    }

    fn y(&self) -> f64 {
        match &self.kind {
            Kind::Point(p) => p.center().latitude() * R2D,
            _ => f64::NAN,
        }
    }
}

impl Geo3dShape {
    fn rect_bounds(&self) -> [f64; 4] {
        match self.kind {
            Kind::Rectangle {
                min_x,
                max_x,
                min_y,
                max_y,
                ..
            } => [min_x, max_x, min_y, max_y],
            _ => [f64::NAN; 4],
        }
    }
}

impl Rectangle for Geo3dShape {
    fn min_x(&self) -> f64 {
        self.rect_bounds()[0]
    }

    fn max_x(&self) -> f64 {
        self.rect_bounds()[1]
    }

    fn min_y(&self) -> f64 {
        self.rect_bounds()[2]
    }

    fn max_y(&self) -> f64 {
        self.rect_bounds()[3]
    }

    fn width(&self) -> f64 {
        let mut result = self.max_x() - self.min_x();
        if result < 0.0 {
            result += 360.0;
        }
        result
    }

    fn height(&self) -> f64 {
        self.max_y() - self.min_y()
    }

    fn crosses_date_line(&self) -> bool {
        self.max_x() > 0.0 && self.min_x() < 0.0
    }

    fn relate_y_range(&self, min_y: f64, max_y: f64) -> Result<SpatialRelation> {
        let r = self.ctx.rect(-180.0, 180.0, min_y, max_y)?;
        self.relate(&*r)
    }

    fn relate_x_range(&self, min_x: f64, max_x: f64) -> Result<SpatialRelation> {
        let r = self.ctx.rect(min_x, max_x, -90.0, 90.0)?;
        self.relate(&*r)
    }
}

impl Circle for Geo3dShape {
    fn radius(&self) -> f64 {
        match &self.kind {
            Kind::Circle(c) => c.radius() * R2D,
            _ => f64::NAN,
        }
    }
}

/// `Geo3dDistanceCalculator`: surface distances on a planet model, in
/// degrees.
#[derive(Debug, Clone)]
pub struct Geo3dDistanceCalculator {
    planet_model: Arc<PlanetModel>,
}

impl Geo3dDistanceCalculator {
    pub fn new(planet_model: Arc<PlanetModel>) -> Self {
        Geo3dDistanceCalculator { planet_model }
    }

    fn geo_point_of(&self, p: &dyn Point) -> Result<GeoPoint> {
        if let Some(ps) = p
            .as_any()
            .downcast_ref::<Geo3dShape>()
            .and_then(|g| g.geo_point_shape())
        {
            return Ok(ps.center());
        }
        Ok(GeoPoint::from_lat_lon(
            &self.planet_model,
            p.y() * D2R,
            p.x() * D2R,
        )?)
    }
}

impl fmt::Display for Geo3dDistanceCalculator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Java prints `Object.toString()`: the class name and an identity
        // hash, which is left out.
        f.write_str("org.apache.lucene.spatial.spatial4j.Geo3dDistanceCalculator")
    }
}

impl DistanceCalculator for Geo3dDistanceCalculator {
    fn distance(&self, from: &dyn Point, to: &dyn Point) -> Result<f64> {
        let pf = from
            .as_any()
            .downcast_ref::<Geo3dShape>()
            .and_then(|g| g.geo_point_shape());
        let pt = to
            .as_any()
            .downcast_ref::<Geo3dShape>()
            .and_then(|g| g.geo_point_shape());
        if let (Some(a), Some(b)) = (pf, pt) {
            return Ok(self.planet_model.surface_distance(&a.center(), &b.center()) * R2D);
        }
        self.distance_xy(from, to.x(), to.y())
    }

    fn distance_xy(&self, from: &dyn Point, to_x: f64, to_y: f64) -> Result<f64> {
        let from_geo_point = self.geo_point_of(from)?;
        let to_geo_point = GeoPoint::from_lat_lon(&self.planet_model, to_y * D2R, to_x * D2R)?;
        Ok(self
            .planet_model
            .surface_distance(&from_geo_point, &to_geo_point)
            * R2D)
    }

    fn within(&self, from: &dyn Point, to_x: f64, to_y: f64, distance: f64) -> Result<bool> {
        Ok(distance < self.distance_xy(from, to_x, to_y)?)
    }

    fn point_on_bearing(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        bearing_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Point>> {
        let Some(ps) = from
            .as_any()
            .downcast_ref::<Geo3dShape>()
            .and_then(|g| g.geo_point_shape())
        else {
            return Err(Error::ClassCast(format!(
                "class {} cannot be cast to class org.apache.lucene.spatial.spatial4j.Geo3dPointShape",
                crate::spatial4j::binary_codec::java_class_name(&**from)
            )));
        };
        let point = ps.center();
        let dist = D2R * dist_deg;
        let bearing = D2R * bearing_deg;
        let new_point = self
            .planet_model
            .surface_point_on_bearing(&point, dist, bearing)?;
        let new_lat = new_point.latitude() * R2D;
        let new_lon = new_point.longitude() * R2D;
        ctx.point_xy(new_lon, new_lat)
    }

    fn calc_box_by_dist_from_pt(
        &self,
        from: &Arc<dyn Point>,
        dist_deg: f64,
        ctx: &Arc<SpatialContext>,
    ) -> Result<Arc<dyn Rectangle>> {
        let circle = ctx.circle_at(from, dist_deg)?;
        circle.bounding_box()
    }

    fn calc_box_by_dist_from_pt_y_horiz_axis_deg(
        &self,
        _from: &dyn Point,
        _dist_deg: f64,
        _ctx: &SpatialContext,
    ) -> Result<f64> {
        Err(Error::UnsupportedOperation(None))
    }

    fn area_rect(&self, _rect: &dyn Rectangle) -> Result<f64> {
        Err(Error::UnsupportedOperation(None))
    }

    fn area_circle(&self, _circle: &dyn Circle) -> Result<f64> {
        Err(Error::UnsupportedOperation(None))
    }

    fn equals(&self, other: &dyn DistanceCalculator) -> bool {
        // Java does not override equals: identity.
        std::ptr::eq(
            other.as_any() as *const dyn Any as *const u8,
            self as *const Self as *const u8,
        )
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// `Geo3dBinaryCodec`: shapes serialized as geo3d serializes them (class,
/// then fields), on the context's planet model.
#[derive(Debug, Clone)]
pub struct Geo3dBinaryCodec {
    planet_model: Arc<PlanetModel>,
}

impl Geo3dBinaryCodec {
    pub fn new(planet_model: Arc<PlanetModel>) -> Self {
        Geo3dBinaryCodec { planet_model }
    }
}

/// The geo3d area shape a deserialized object is, if it is one.
fn as_area_shape(object: StandardObject) -> std::result::Result<Arc<dyn GeoAreaShape>, String> {
    match object {
        StandardObject::Polygon(s) => Ok(s),
        StandardObject::PointShape(s) => Ok(s),
        StandardObject::BBox(s) => Ok(s),
        StandardObject::Circle(s) => Ok(s),
        StandardObject::Path(s) => Ok(s),
        StandardObject::AreaShape(s) => Ok(s),
        other => Err(other.class_name().to_string()),
    }
}

impl BinaryCodec for Geo3dBinaryCodec {
    fn read_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        input: &mut DataInput<'_>,
    ) -> Result<Arc<dyn Shape>> {
        let bytes = input.remaining();
        let mut geo_input = crate::spatial3d::serializable::Input::new(bytes);
        let object = read_object(&self.planet_model, &mut geo_input)?;
        input.advance(bytes.len() - geo_input.remaining());
        match as_area_shape(object) {
            Ok(shape) => Ok(Arc::new(Geo3dShape::new(shape, ctx.clone()))),
            Err(class) => Err(Error::IllegalArgument(format!(
                "trying to read a not supported shape: class org.apache.lucene.spatial3d.geom.{class}"
            ))),
        }
    }

    fn write_shape(&self, out: &mut Vec<u8>, s: &dyn Shape) -> Result<()> {
        match s.as_any().downcast_ref::<Geo3dShape>() {
            Some(g) => Ok(write_object(out, &*g.shape)?),
            None => Err(Error::IllegalArgument(format!(
                "trying to write a not supported shape: {}",
                crate::spatial4j::binary_codec::java_class_name(s)
            ))),
        }
    }
}

/// `Geo3dShapeFactory`.
#[derive(Debug, Clone)]
pub struct Geo3dShapeFactory {
    norm_wrap_longitude: bool,
    planet_model: Arc<PlanetModel>,
    circle_accuracy: f64,
}

/// `DEFAULT_CIRCLE_ACCURACY`.
const DEFAULT_CIRCLE_ACCURACY: f64 = 1e-4;

impl Geo3dShapeFactory {
    /// `new Geo3dShapeFactory(context, factory)`.
    pub fn new(planet_model: Arc<PlanetModel>, norm_wrap_longitude: bool) -> Self {
        Geo3dShapeFactory {
            norm_wrap_longitude,
            planet_model,
            circle_accuracy: DEFAULT_CIRCLE_ACCURACY,
        }
    }

    /// `setCircleAccuracy(circleAccuracy)`.
    pub fn set_circle_accuracy(&mut self, circle_accuracy: f64) {
        self.circle_accuracy = circle_accuracy;
    }

    /// The planet model shapes are made on.
    pub fn planet_model(&self) -> &Arc<PlanetModel> {
        &self.planet_model
    }

    fn geo_point(&self, point: &S2Point) -> GeoPoint {
        self.planet_model
            .create_surface_point_xyz(point.x, point.y, point.z)
    }
}

impl ShapeFactory for Geo3dShapeFactory {
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
        let point = make_geo_point_shape(&self.planet_model, y * D2R, x * D2R)?;
        Ok(Arc::new(Geo3dShape::new_point(point, ctx.clone())))
    }

    fn point_xy_unchecked(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
    ) -> Result<Arc<dyn Point>> {
        self.point_xy(ctx, x, y)
    }

    fn point_xyz(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<Arc<dyn Point>> {
        let point = GeoPoint::new(x, y, z);
        let shape = make_geo_point_shape(&self.planet_model, point.latitude(), point.longitude())?;
        Ok(Arc::new(Geo3dShape::new_point(shape, ctx.clone())))
    }

    fn rect(
        &self,
        ctx: &Arc<SpatialContext>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Result<Arc<dyn Rectangle>> {
        let bbox = make_geo_bbox(
            &self.planet_model,
            max_y * D2R,
            min_y * D2R,
            min_x * D2R,
            max_x * D2R,
        )?;
        Ok(Arc::new(Geo3dShape::new_rectangle_with_bounds(
            bbox,
            ctx.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
        )))
    }

    fn circle(
        &self,
        ctx: &Arc<SpatialContext>,
        x: f64,
        y: f64,
        distance: f64,
    ) -> Result<Arc<dyn Circle>> {
        let circle = if self.planet_model.is_sphere() {
            make_geo_circle(&self.planet_model, y * D2R, x * D2R, distance * D2R)?
        } else {
            make_exact_geo_circle(
                &self.planet_model,
                y * D2R,
                x * D2R,
                distance * D2R,
                self.circle_accuracy * D2R,
            )?
        };
        Ok(Arc::new(Geo3dShape::new_circle(circle, ctx.clone())))
    }

    fn circle_at(
        &self,
        ctx: &Arc<SpatialContext>,
        point: &Arc<dyn Point>,
        distance: f64,
    ) -> Result<Arc<dyn Circle>> {
        self.circle(ctx, point.x(), point.y(), distance)
    }

    fn line_string(
        &self,
        ctx: &Arc<SpatialContext>,
        points: &[Arc<dyn Point>],
        buf: f64,
    ) -> Result<Arc<dyn Shape>> {
        let mut builder = self.line_string_builder(ctx);
        for point in points {
            builder.point_xy(point.x(), point.y())?;
        }
        builder.buffer(buf);
        builder.build()
    }

    fn multi_shape(
        &self,
        _ctx: &Arc<SpatialContext>,
        _coll: Vec<Arc<dyn Shape>>,
    ) -> Result<ShapeCollection> {
        Err(Error::UnsupportedOperation(None))
    }

    fn line_string_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn LineStringBuilder> {
        Box::new(Geo3dLineStringBuilder {
            points: Geo3dPoints::new(self.planet_model.clone()),
            distance: 0.0,
            ctx: ctx.clone(),
        })
    }

    fn polygon_builder(&self, ctx: &Arc<SpatialContext>) -> Result<Box<dyn PolygonBuilder>> {
        Ok(Box::new(Geo3dPolygonBuilder {
            points: Geo3dPoints::new(self.planet_model.clone()),
            poly_holes: Vec::new(),
            ctx: ctx.clone(),
        }))
    }

    fn multi_shape_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiShapeBuilder> {
        Box::new(Geo3dMultiShapeBuilder {
            composite: GeoCompositeAreaShape::new(&self.planet_model),
            ctx: ctx.clone(),
        })
    }

    fn multi_point_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPointBuilder> {
        Box::new(Geo3dMultiPointBuilder {
            points: Geo3dPoints::new(self.planet_model.clone()),
            ctx: ctx.clone(),
        })
    }

    fn multi_line_string_builder(
        &self,
        ctx: &Arc<SpatialContext>,
    ) -> Box<dyn MultiLineStringBuilder> {
        Box::new(Geo3dMultiLineBuilder {
            builders: Vec::new(),
            factory: self.clone(),
            ctx: ctx.clone(),
        })
    }

    fn multi_polygon_builder(&self, ctx: &Arc<SpatialContext>) -> Box<dyn MultiPolygonBuilder> {
        Box::new(Geo3dMultiPolygonBuilder {
            builders: Vec::new(),
            factory: self.clone(),
            ctx: ctx.clone(),
        })
    }

    fn as_s2(&self) -> Option<&dyn S2ShapeFactory> {
        Some(self)
    }
}

impl S2ShapeFactory for Geo3dShapeFactory {
    fn s2_cell_shape(
        &self,
        ctx: &Arc<SpatialContext>,
        cell_id: S2CellId,
    ) -> Result<Arc<dyn Shape>> {
        let cell = S2Cell::new(cell_id);
        let point1 = self.geo_point(&cell.vertex_raw(0));
        let point2 = self.geo_point(&cell.vertex_raw(1));
        let point3 = self.geo_point(&cell.vertex_raw(2));
        let point4 = self.geo_point(&cell.vertex_raw(3));
        let shape = make_geo_s2_shape(&self.planet_model, point1, point2, point3, point4)?;
        Ok(Arc::new(Geo3dShape::new(shape, ctx.clone())))
    }
}

/// `Geo3dPointBuilder`'s point list.
#[derive(Debug, Clone)]
struct Geo3dPoints {
    planet_model: Arc<PlanetModel>,
    points: Vec<GeoPoint>,
}

impl Geo3dPoints {
    fn new(planet_model: Arc<PlanetModel>) -> Self {
        Geo3dPoints {
            planet_model,
            points: Vec::new(),
        }
    }
}

impl PointsBuilder for Geo3dPoints {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        let point = GeoPoint::from_lat_lon(&self.planet_model, y * D2R, x * D2R)?;
        self.points.push(point);
        Ok(())
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        let point = GeoPoint::new(x, y, z);
        // `List.contains`: `GeoPoint.equals` compares x, y and z.
        if !self
            .points
            .iter()
            .any(|p| p.x == point.x && p.y == point.y && p.z == point.z)
        {
            self.points.push(point);
        }
        Ok(())
    }
}

/// `Geo3dLineStringBuilder`: a geo3d path (`distance` is its cutoff angle,
/// passed through as is).
struct Geo3dLineStringBuilder {
    points: Geo3dPoints,
    distance: f64,
    ctx: Arc<SpatialContext>,
}

impl PointsBuilder for Geo3dLineStringBuilder {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.points.point_xy(x, y)
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.points.point_xyz(x, y, z)
    }
}

impl Geo3dLineStringBuilder {
    fn build_area(&self) -> Result<Arc<dyn GeoAreaShape>> {
        let path = make_geo_path(
            &self.points.planet_model,
            self.distance,
            &self.points.points,
        )?;
        Ok(path)
    }
}

impl LineStringBuilder for Geo3dLineStringBuilder {
    fn buffer(&mut self, distance: f64) {
        self.distance = distance;
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(Geo3dShape::new(
            self.build_area()?,
            self.ctx.clone(),
        )))
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

/// `Geo3dPolygonBuilder`.
struct Geo3dPolygonBuilder {
    points: Geo3dPoints,
    poly_holes: Vec<PolygonDescription>,
    ctx: Arc<SpatialContext>,
}

impl PointsBuilder for Geo3dPolygonBuilder {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.points.point_xy(x, y)
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.points.point_xyz(x, y, z)
    }
}

impl Geo3dPolygonBuilder {
    fn build_area(&self) -> Result<Arc<dyn GeoAreaShape>> {
        let description =
            PolygonDescription::with_holes(self.points.points.clone(), self.poly_holes.clone());
        match make_geo_polygon_from_description(&self.points.planet_model, &description)? {
            Some(p) => Ok(p),
            None => Err(Error::InvalidShape(
                "Invalid polygon, all points are coplanar".into(),
            )),
        }
    }
}

/// `Geo3dPolygonBuilder.Geo3dHoleBuilder`.
struct Geo3dHoleBuilder<'a> {
    points: Geo3dPoints,
    polygon: &'a mut Geo3dPolygonBuilder,
}

impl PointsBuilder for Geo3dHoleBuilder<'_> {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.points.point_xy(x, y)
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.points.point_xyz(x, y, z)
    }
}

impl HoleBuilder for Geo3dHoleBuilder<'_> {
    fn end_hole(self: Box<Self>) -> Result<()> {
        let this = *self;
        this.polygon
            .poly_holes
            .push(PolygonDescription::new(this.points.points));
        Ok(())
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

impl PolygonBuilder for Geo3dPolygonBuilder {
    fn hole(&mut self) -> Box<dyn HoleBuilder + '_> {
        let pm = self.points.planet_model.clone();
        Box::new(Geo3dHoleBuilder {
            points: Geo3dPoints::new(pm),
            polygon: self,
        })
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(Geo3dShape::new(
            self.build_area()?,
            self.ctx.clone(),
        )))
    }

    fn build_or_rect(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        self.build()
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

/// `Geo3dMultiPointBuilder`: a composite of point shapes.
struct Geo3dMultiPointBuilder {
    points: Geo3dPoints,
    ctx: Arc<SpatialContext>,
}

impl PointsBuilder for Geo3dMultiPointBuilder {
    fn point_xy(&mut self, x: f64, y: f64) -> Result<()> {
        self.points.point_xy(x, y)
    }

    fn point_xyz(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.points.point_xyz(x, y, z)
    }
}

impl MultiPointBuilder for Geo3dMultiPointBuilder {
    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        let pm = &self.points.planet_model;
        let mut area = GeoCompositeAreaShape::new(pm);
        for point in &self.points.points {
            let shape = make_geo_point_shape(pm, point.latitude(), point.longitude())?;
            area.add_shape(shape)?;
        }
        Ok(Arc::new(Geo3dShape::new(Arc::new(area), self.ctx.clone())))
    }

    fn as_points_builder(&mut self) -> &mut dyn PointsBuilder {
        self
    }
}

/// `Geo3dMultiLineBuilder`.
struct Geo3dMultiLineBuilder {
    builders: Vec<Box<dyn LineStringBuilder>>,
    factory: Geo3dShapeFactory,
    ctx: Arc<SpatialContext>,
}

impl MultiLineStringBuilder for Geo3dMultiLineBuilder {
    fn line_string(&self) -> Box<dyn LineStringBuilder> {
        self.factory.line_string_builder(&self.ctx)
    }

    fn add(&mut self, builder: Box<dyn LineStringBuilder>) -> Result<()> {
        self.builders.push(builder);
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        let mut area = GeoCompositeAreaShape::new(&self.factory.planet_model);
        for builder in self.builders {
            let shape = builder.build()?;
            let inner = geo3d_area(&*shape).expect("a Geo3D line string").clone();
            area.add_shape(inner)?;
        }
        Ok(Arc::new(Geo3dShape::new(Arc::new(area), self.ctx.clone())))
    }
}

/// `Geo3dMultiPolygonBuilder`.
struct Geo3dMultiPolygonBuilder {
    builders: Vec<Box<dyn PolygonBuilder>>,
    factory: Geo3dShapeFactory,
    ctx: Arc<SpatialContext>,
}

impl MultiPolygonBuilder for Geo3dMultiPolygonBuilder {
    fn polygon(&self) -> Result<Box<dyn PolygonBuilder>> {
        self.factory.polygon_builder(&self.ctx)
    }

    fn add(&mut self, builder: Box<dyn PolygonBuilder>) -> Result<()> {
        self.builders.push(builder);
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        let mut area = GeoCompositeAreaShape::new(&self.factory.planet_model);
        for builder in self.builders {
            let shape = builder.build()?;
            let inner = geo3d_area(&*shape).expect("a Geo3D polygon").clone();
            area.add_shape(inner)?;
        }
        Ok(Arc::new(Geo3dShape::new(Arc::new(area), self.ctx.clone())))
    }
}

/// `Geo3dMultiShapeBuilder`: a composite of Geo3D shapes.
struct Geo3dMultiShapeBuilder {
    composite: GeoCompositeAreaShape,
    ctx: Arc<SpatialContext>,
}

impl MultiShapeBuilder for Geo3dMultiShapeBuilder {
    fn add(&mut self, shape: Arc<dyn Shape>) -> Result<()> {
        let Some(area) = geo3d_area(&*shape) else {
            return Err(Error::ClassCast(format!(
                "class {} cannot be cast to class org.apache.lucene.spatial.spatial4j.Geo3dShape",
                crate::spatial4j::binary_codec::java_class_name(&*shape)
            )));
        };
        self.composite.add_shape(area.clone())?;
        Ok(())
    }

    fn build(self: Box<Self>) -> Result<Arc<dyn Shape>> {
        Ok(Arc::new(Geo3dShape::new(
            Arc::new(self.composite),
            self.ctx.clone(),
        )))
    }
}

#[cfg(test)]
#[path = "spatial4j_tests.rs"]
mod tests;
