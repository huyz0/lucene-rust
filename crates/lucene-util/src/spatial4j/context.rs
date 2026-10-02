//! `SpatialContext` and `SpatialContextFactory`
//! (`org.locationtech.spatial4j.context`), with Lucene's
//! `Geo3dSpatialContextFactory` as a factory kind.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Arc, OnceLock};

use super::binary_codec::{BinaryCodec, DefaultBinaryCodec};
use super::collection::ShapeCollection;
use super::distance::{CartesianDistCalc, DistanceCalculator, GeodesicSphereDistCalc};
use super::rectangle::RectangleImpl;
use super::shape::{Circle, Point, Rectangle, Shape};
use super::shape_factory::{ShapeFactory, ShapeFactoryImpl};
use super::wkt::WktReader;
use super::{Error, Result};
use crate::spatial3d::PlanetModel;
use crate::spatial_extras::spatial4j as geo3d;

/// Which `SpatialContextFactory` class a factory stands for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FactoryKind {
    /// Spatial4j's own `SpatialContextFactory`: `ShapeFactoryImpl` and
    /// `BinaryCodec`.
    Spatial4j,
    /// Lucene's `Geo3dSpatialContextFactory`: `Geo3dShapeFactory`,
    /// `Geo3dBinaryCodec` and (by default) `Geo3dDistanceCalculator` on a
    /// `PlanetModel`.
    Geo3d,
}

/// `SpatialContextFactory` (and `Geo3dSpatialContextFactory`): the
/// configuration a [`SpatialContext`] is built from.
#[derive(Clone)]
pub struct SpatialContextFactory {
    /// Which factory class this is.
    pub kind: FactoryKind,
    /// `geo`: geodetic (degrees on a sphere) or planar.
    pub geo: bool,
    /// `distCalc`; `None` picks the default in [`Self::new_spatial_context`].
    pub dist_calc: Option<Arc<dyn DistanceCalculator>>,
    /// `worldBounds` as `[minX, maxX, minY, maxY]`; `None` picks the
    /// default.
    pub world_bounds: Option<[f64; 4]>,
    /// `normWrapLongitude`.
    pub norm_wrap_longitude: bool,
    /// `Geo3dSpatialContextFactory.planetModel` (the Geo3D kind only).
    pub planet_model: Option<Arc<PlanetModel>>,
}

impl fmt::Debug for SpatialContextFactory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SpatialContextFactory")
            .field("kind", &self.kind)
            .field("geo", &self.geo)
            .field("dist_calc", &self.dist_calc.as_ref().map(|c| c.to_string()))
            .field("world_bounds", &self.world_bounds)
            .field("norm_wrap_longitude", &self.norm_wrap_longitude)
            .finish()
    }
}

impl Default for SpatialContextFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl SpatialContextFactory {
    /// `new SpatialContextFactory()`: geo, every other setting defaulted.
    pub fn new() -> Self {
        SpatialContextFactory {
            kind: FactoryKind::Spatial4j,
            geo: true,
            dist_calc: None,
            world_bounds: None,
            norm_wrap_longitude: false,
            planet_model: None,
        }
    }

    /// `new Geo3dSpatialContextFactory()`.
    pub fn geo3d() -> Self {
        SpatialContextFactory {
            kind: FactoryKind::Geo3d,
            ..Self::new()
        }
    }

    /// `SpatialContextFactory.makeSpatialContext(args, classLoader)`: the
    /// factory class from `spatialContextFactory`, then each setting from
    /// its key (`geo`, `distCalculator`, `worldBounds`, `normWrapLongitude`,
    /// and for Geo3D `planetModel`).
    ///
    /// Only the two factory classes this port has are accepted, and
    /// `shapeFactoryClass`/`binaryCodecClass`/`readers`/`writers` only name
    /// the classes the factory would pick anyway; `worldBounds` is read as
    /// WKT (Java tries every registered reader).
    pub fn make_spatial_context(args: &BTreeMap<String, String>) -> Result<Arc<SpatialContext>> {
        let mut instance = match args.get("spatialContextFactory").map(String::as_str) {
            None | Some("org.locationtech.spatial4j.context.SpatialContextFactory") => {
                SpatialContextFactory::new()
            }
            Some("org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory") => {
                SpatialContextFactory::geo3d()
            }
            Some(other) => {
                return Err(Error::Runtime(format!(
                    "java.lang.ClassNotFoundException: {other}"
                )))
            }
        };
        instance.init(args)?;
        instance.new_spatial_context()
    }

    /// `init(args, classLoader)`.
    fn init(&mut self, args: &BTreeMap<String, String>) -> Result<()> {
        if self.kind == FactoryKind::Geo3d {
            self.init_planet_model(args)?;
        }
        if let Some(v) = args.get("geo") {
            self.geo = v.eq_ignore_ascii_case("true");
        }
        self.check_class_field(args, "shapeFactoryClass")?;
        self.init_calculator(args)?;
        if let Some(wb) = args.get("worldBounds") {
            // Java: kinda ugly we do this just to read a rectangle.
            let ctx = self.clone().new_spatial_context()?;
            let shape = ctx.read_shape_from_wkt(wb)?;
            let r = shape.as_rectangle().ok_or_else(|| {
                Error::ClassCast(format!(
                    "{shape} is not a org.locationtech.spatial4j.shape.Rectangle"
                ))
            })?;
            self.world_bounds = Some([r.min_x(), r.max_x(), r.min_y(), r.max_y()]);
        }
        if let Some(v) = args.get("normWrapLongitude") {
            self.norm_wrap_longitude = v.eq_ignore_ascii_case("true");
        }
        self.check_class_field(args, "binaryCodecClass")
    }

    /// A `Class`-typed setting naming the class the factory already uses.
    fn check_class_field(&self, args: &BTreeMap<String, String>, name: &str) -> Result<()> {
        let Some(v) = args.get(name) else {
            return Ok(());
        };
        let expected = match (name, self.kind) {
            ("shapeFactoryClass", FactoryKind::Spatial4j) => {
                "org.locationtech.spatial4j.shape.impl.ShapeFactoryImpl"
            }
            ("shapeFactoryClass", FactoryKind::Geo3d) => {
                "org.apache.lucene.spatial.spatial4j.Geo3dShapeFactory"
            }
            (_, FactoryKind::Spatial4j) => "org.locationtech.spatial4j.io.BinaryCodec",
            (_, FactoryKind::Geo3d) => "org.apache.lucene.spatial.spatial4j.Geo3dBinaryCodec",
        };
        if v == expected {
            Ok(())
        } else {
            Err(Error::Runtime(format!(
                "Invalid value '{v}' on field {name} of type class java.lang.Class"
            )))
        }
    }

    /// `initCalculator()` (and `Geo3dSpatialContextFactory`'s override,
    /// which adds `geo3d`).
    fn init_calculator(&mut self, args: &BTreeMap<String, String>) -> Result<()> {
        let Some(calc) = args.get("distCalculator") else {
            return Ok(());
        };
        if self.kind == FactoryKind::Geo3d && calc == "geo3d" {
            let pm = self
                .planet_model
                .clone()
                .unwrap_or_else(PlanetModel::sphere);
            self.dist_calc = Some(Arc::new(geo3d::Geo3dDistanceCalculator::new(pm)));
            return Ok(());
        }
        self.dist_calc = Some(if calc.eq_ignore_ascii_case("haversine") {
            Arc::new(GeodesicSphereDistCalc::Haversine)
        } else if calc.eq_ignore_ascii_case("lawOfCosines") {
            Arc::new(GeodesicSphereDistCalc::LawOfCosines)
        } else if calc.eq_ignore_ascii_case("vincentySphere") {
            Arc::new(GeodesicSphereDistCalc::Vincenty)
        } else if calc.eq_ignore_ascii_case("cartesian") {
            Arc::new(CartesianDistCalc::new(false))
        } else if calc.eq_ignore_ascii_case("cartesian^2") {
            Arc::new(CartesianDistCalc::new(true))
        } else {
            return Err(Error::Runtime(format!("Unknown calculator: {calc}")));
        });
        Ok(())
    }

    /// `Geo3dSpatialContextFactory.initPlanetModel(args)`.
    fn init_planet_model(&mut self, args: &BTreeMap<String, String>) -> Result<()> {
        self.planet_model = Some(match args.get("planetModel") {
            None => PlanetModel::sphere(),
            Some(pm) if pm.eq_ignore_ascii_case("sphere") => PlanetModel::sphere(),
            Some(pm) if pm.eq_ignore_ascii_case("wgs84") => PlanetModel::wgs84(),
            Some(pm) if pm.eq_ignore_ascii_case("clarke1866") => PlanetModel::clarke_1866(),
            Some(pm) => return Err(Error::Runtime(format!("Unknown planet model: {pm}"))),
        });
        Ok(())
    }

    /// `newSpatialContext()`.
    pub fn new_spatial_context(mut self) -> Result<Arc<SpatialContext>> {
        if self.kind == FactoryKind::Geo3d {
            let pm = self
                .planet_model
                .get_or_insert_with(PlanetModel::sphere)
                .clone();
            if self.dist_calc.is_none() {
                self.dist_calc = Some(Arc::new(geo3d::Geo3dDistanceCalculator::new(pm)));
            }
        }
        SpatialContext::new(&self).map(Arc::new)
    }
}

/// `SpatialContext`: whether the world is geodetic, its bounds, and the
/// shape factory, distance calculator and binary codec every shape and
/// strategy uses.
pub struct SpatialContext {
    geo: bool,
    shape_factory: Box<dyn ShapeFactory>,
    calculator: Arc<dyn DistanceCalculator>,
    world_bounds: [f64; 4],
    binary_codec: Box<dyn BinaryCodec>,
}

impl fmt::Debug for SpatialContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

static GEO: OnceLock<Arc<SpatialContext>> = OnceLock::new();

impl SpatialContext {
    /// `SpatialContext.GEO`: the default geodetic context (haversine).
    pub fn geo_context() -> Arc<SpatialContext> {
        GEO.get_or_init(|| {
            SpatialContextFactory::new()
                .new_spatial_context()
                .expect("the default geo context is valid")
        })
        .clone()
    }

    /// `new SpatialContext(factory)`.
    pub fn new(factory: &SpatialContextFactory) -> Result<SpatialContext> {
        let geo = factory.geo;
        let shape_factory: Box<dyn ShapeFactory> = match factory.kind {
            FactoryKind::Spatial4j => {
                Box::new(ShapeFactoryImpl::new(geo && factory.norm_wrap_longitude))
            }
            FactoryKind::Geo3d => Box::new(geo3d::Geo3dShapeFactory::new(
                factory
                    .planet_model
                    .clone()
                    .unwrap_or_else(PlanetModel::sphere),
                geo && factory.norm_wrap_longitude,
            )),
        };
        let calculator: Arc<dyn DistanceCalculator> = match &factory.dist_calc {
            Some(c) => c.clone(),
            None if geo => Arc::new(GeodesicSphereDistCalc::Haversine),
            None => Arc::new(CartesianDistCalc::new(false)),
        };
        let world_bounds = match factory.world_bounds {
            None if geo => [-180.0, 180.0, -90.0, 90.0],
            None => [-f64::MAX, f64::MAX, -f64::MAX, f64::MAX],
            Some(b) => {
                let show = |b: [f64; 4]| {
                    format!(
                        "Rect(minX={},maxX={},minY={},maxY={})",
                        super::dstr(b[0]),
                        super::dstr(b[1]),
                        super::dstr(b[2]),
                        super::dstr(b[3])
                    )
                };
                let geo_bounds = [-180.0, 180.0, -90.0, 90.0];
                if geo
                    && !b
                        .iter()
                        .zip(geo_bounds.iter())
                        .all(|(&a, &g)| super::double_compare_eq(a, g))
                {
                    return Err(Error::IllegalArgument(format!(
                        "for geo (lat/lon), bounds must be {}",
                        show(geo_bounds)
                    )));
                }
                if b[0] > b[1] {
                    return Err(Error::IllegalArgument(format!(
                        "worldBounds minX should be <= maxX: {}",
                        show(b)
                    )));
                }
                if b[2] > b[3] {
                    return Err(Error::IllegalArgument(format!(
                        "worldBounds minY should be <= maxY: {}",
                        show(b)
                    )));
                }
                b
            }
        };
        let binary_codec: Box<dyn BinaryCodec> = match factory.kind {
            FactoryKind::Spatial4j => Box::new(DefaultBinaryCodec),
            FactoryKind::Geo3d => Box::new(geo3d::Geo3dBinaryCodec::new(
                factory
                    .planet_model
                    .clone()
                    .unwrap_or_else(PlanetModel::sphere),
            )),
        };
        Ok(SpatialContext {
            geo,
            shape_factory,
            calculator,
            world_bounds,
            binary_codec,
        })
    }

    /// `getShapeFactory()`.
    pub fn shape_factory(&self) -> &dyn ShapeFactory {
        &*self.shape_factory
    }

    /// `getDistCalc()`.
    pub fn dist_calc(&self) -> &Arc<dyn DistanceCalculator> {
        &self.calculator
    }

    /// `calcDistance(p, x2, y2)`.
    pub fn calc_distance(&self, p: &dyn Point, x2: f64, y2: f64) -> Result<f64> {
        self.calculator.distance_xy(p, x2, y2)
    }

    /// `calcDistance(p, p2)`.
    pub fn calc_distance_pts(&self, p: &dyn Point, p2: &dyn Point) -> Result<f64> {
        self.calculator.distance(p, p2)
    }

    /// `getWorldBounds()`.
    pub fn world_bounds(self: &Arc<Self>) -> RectangleImpl {
        let [min_x, max_x, min_y, max_y] = self.world_bounds;
        RectangleImpl::new(min_x, max_x, min_y, max_y, self.clone())
    }

    /// The world bounds as `[minX, maxX, minY, maxY]`.
    pub fn world_bounds_values(&self) -> [f64; 4] {
        self.world_bounds
    }

    /// `isNormWrapLongitude()`.
    pub fn is_norm_wrap_longitude(&self) -> bool {
        self.shape_factory.is_norm_wrap_longitude()
    }

    /// `isGeo()`.
    pub fn is_geo(&self) -> bool {
        self.geo
    }

    /// `normX(x)`.
    pub fn norm_x(&self, x: f64) -> f64 {
        self.shape_factory.norm_x(x)
    }

    /// `normY(y)`.
    pub fn norm_y(&self, y: f64) -> f64 {
        self.shape_factory.norm_y(y)
    }

    /// `verifyX(x)`.
    pub fn verify_x(self: &Arc<Self>, x: f64) -> Result<()> {
        self.shape_factory.verify_x(self, x)
    }

    /// `verifyY(y)`.
    pub fn verify_y(self: &Arc<Self>, y: f64) -> Result<()> {
        self.shape_factory.verify_y(self, y)
    }

    /// `makePoint(x, y)` / `getShapeFactory().pointXY(x, y)`.
    pub fn point_xy(self: &Arc<Self>, x: f64, y: f64) -> Result<Arc<dyn Point>> {
        self.shape_factory.point_xy(self, x, y)
    }

    /// `makeRectangle(minX, maxX, minY, maxY)` /
    /// `getShapeFactory().rect(..)`.
    pub fn rect(
        self: &Arc<Self>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Result<Arc<dyn Rectangle>> {
        self.shape_factory.rect(self, min_x, max_x, min_y, max_y)
    }

    /// `makeRectangle(lowerLeft, upperRight)`.
    pub fn rect_from_points(
        self: &Arc<Self>,
        lower_left: &dyn Point,
        upper_right: &dyn Point,
    ) -> Result<Arc<dyn Rectangle>> {
        self.rect(
            lower_left.x(),
            upper_right.x(),
            lower_left.y(),
            upper_right.y(),
        )
    }

    /// `makeCircle(x, y, distance)`.
    pub fn circle(self: &Arc<Self>, x: f64, y: f64, distance: f64) -> Result<Arc<dyn Circle>> {
        self.shape_factory.circle(self, x, y, distance)
    }

    /// `makeCircle(point, distance)`.
    pub fn circle_at(
        self: &Arc<Self>,
        point: &Arc<dyn Point>,
        distance: f64,
    ) -> Result<Arc<dyn Circle>> {
        self.shape_factory.circle_at(self, point, distance)
    }

    /// `makeLineString(points)` / `makeBufferedLineString(points, buf)`.
    pub fn line_string(
        self: &Arc<Self>,
        points: &[Arc<dyn Point>],
        buf: f64,
    ) -> Result<Arc<dyn Shape>> {
        self.shape_factory.line_string(self, points, buf)
    }

    /// `makeCollection(shapes)`.
    pub fn collection(self: &Arc<Self>, shapes: Vec<Arc<dyn Shape>>) -> Result<ShapeCollection> {
        self.shape_factory.multi_shape(self, shapes)
    }

    /// `getFormats().getWktReader()`.
    pub fn wkt_reader(self: &Arc<Self>) -> WktReader {
        WktReader::new(self.clone())
    }

    /// `readShapeFromWkt(wkt)`.
    pub fn read_shape_from_wkt(self: &Arc<Self>, wkt: &str) -> Result<Arc<dyn Shape>> {
        self.wkt_reader().parse(wkt)
    }

    /// `getBinaryCodec()`.
    pub fn binary_codec(&self) -> &dyn BinaryCodec {
        &*self.binary_codec
    }

    /// Whether this is [`SpatialContext::geo_context`] itself (Java's
    /// `equals`, which is identity).
    pub fn is_geo_singleton(self: &Arc<Self>) -> bool {
        GEO.get().is_some_and(|g| Arc::ptr_eq(g, self))
    }
}

impl fmt::Display for SpatialContext {
    /// `toString()`. Java prints `SpatialContext.GEO` for the singleton;
    /// that needs the `Arc` ([`SpatialContext::is_geo_singleton`]), so this
    /// always prints the long form.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let [a, b, c, d] = self.world_bounds;
        write!(
            f,
            "SpatialContext{{geo={}, calculator={}, worldBounds=Rect(minX={},maxX={},minY={},maxY={})}}",
            self.geo,
            self.calculator,
            super::dstr(a),
            super::dstr(b),
            super::dstr(c),
            super::dstr(d)
        )
    }
}
