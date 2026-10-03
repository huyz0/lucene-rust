//! `StandardObjects` (`org.apache.lucene.spatial3d.geom.StandardObjects`) and
//! `SerializableObject`'s object-level helpers: the registry of class codes
//! `writeClass` writes instead of a class name, `writeObject` and
//! `writePlanetObject`.

use std::sync::Arc;

use super::geo_point::GeoPoint;
use super::planet_model::PlanetModel;
use super::serializable::{
    read_boolean, read_string, write_boolean, write_int, write_string, Input,
};
use super::shape::{
    GeoAreaShape, GeoBBox, GeoCircle, GeoMembershipShape, GeoPath, GeoPointShape, GeoPolygon,
    PlanetObject, SerializableObject,
};
use super::xyz_solid::XYZSolid;
use super::{Error, Result};

/// `StandardObjects.CLASS_REGISTRY`: code -> class (simple name; the
/// package is `org.apache.lucene.spatial3d.geom`).
pub const CLASS_NAMES: [&str; 39] = [
    "GeoPoint",
    "GeoRectangle",
    "GeoStandardCircle",
    "GeoStandardPath",
    "GeoConvexPolygon",
    "GeoConcavePolygon",
    "GeoComplexPolygon",
    "GeoCompositePolygon",
    "GeoCompositeMembershipShape",
    "GeoCompositeAreaShape",
    "GeoDegeneratePoint",
    "GeoDegenerateHorizontalLine",
    "GeoDegenerateLatitudeZone",
    "GeoDegenerateLongitudeSlice",
    "GeoDegenerateVerticalLine",
    "GeoLatitudeZone",
    "GeoLongitudeSlice",
    "GeoNorthLatitudeZone",
    "GeoNorthRectangle",
    "GeoSouthLatitudeZone",
    "GeoSouthRectangle",
    "GeoWideDegenerateHorizontalLine",
    "GeoWideLongitudeSlice",
    "GeoWideNorthRectangle",
    "GeoWideRectangle",
    "GeoWideSouthRectangle",
    "GeoWorld",
    "dXdYdZSolid",
    "dXdYZSolid",
    "dXYdZSolid",
    "dXYZSolid",
    "XdYdZSolid",
    "XdYZSolid",
    "XYdZSolid",
    "StandardXYZSolid",
    "PlanetModel",
    "GeoDegeneratePath",
    "GeoExactCircle",
    "GeoS2Shape",
];

/// The `org.apache.lucene.spatial3d.geom` package every registered class is in.
pub const PACKAGE: &str = "org.apache.lucene.spatial3d.geom";

/// The simple class name for a registry code.
pub fn class_name(code: u8) -> Option<&'static str> {
    CLASS_NAMES.get(usize::from(code)).copied()
}

/// `writeClass(outputStream, clazz)`: `true` and the registry code, or
/// `false` and the class name for an unregistered class (none here: every
/// serializable class is registered).
pub fn write_class(out: &mut Vec<u8>, code: Option<u8>, class_name: &str) {
    match code {
        Some(code) => {
            write_boolean(out, true);
            out.push(code);
        }
        None => {
            write_boolean(out, false);
            write_string(out, class_name);
        }
    }
}

/// `writeObject(outputStream, object)`: the class, then the object.
pub fn write_object(out: &mut Vec<u8>, object: &(impl SerializableObject + ?Sized)) -> Result<()> {
    write_class(out, object.class_code(), "");
    object.write(out)
}

/// `writePlanetObject(outputStream, object)`: the planet model, then the
/// object with its class.
pub fn write_planet_object(out: &mut Vec<u8>, object: &(impl PlanetObject + ?Sized)) -> Result<()> {
    object.planet_model().write(out);
    write_object(out, object)
}

/// `writeHomogeneousArray`/`writeHeterogeneousArray`'s count prefix.
pub(crate) fn write_count(out: &mut Vec<u8>, n: usize) {
    write_int(out, n as i32);
}

/// `writePointArray(outputStream, values)`: a homogeneous array -- the count,
/// then each point's fields (no class).
pub fn write_point_array(out: &mut Vec<u8>, values: &[GeoPoint]) {
    write_count(out, values.len());
    for v in values {
        v.write(out);
    }
}

/// `readPointArray(planetModel, inputStream)`.
pub fn read_point_array(input: &mut Input<'_>) -> Result<Vec<GeoPoint>> {
    let count = super::serializable::read_count(input)?;
    // Grown as read: a count off the stream must not size an allocation.
    let mut rval = Vec::new();
    for _ in 0..count {
        rval.push(GeoPoint::read(input)?);
    }
    Ok(rval)
}

/// What `readObject` returned, by its most specific geo3d interface (Java
/// returns a `SerializableObject` and the caller casts).
#[derive(Clone)]
pub enum StandardObject {
    /// `GeoPoint` (not a `PlanetObject`).
    Point(GeoPoint),
    /// An `XYZSolid`.
    Solid(Arc<dyn XYZSolid>),
    /// A `GeoPolygon`.
    Polygon(Arc<dyn GeoPolygon>),
    /// A `GeoPointShape` (`GeoDegeneratePoint`).
    PointShape(Arc<dyn GeoPointShape>),
    /// A `GeoBBox`.
    BBox(Arc<dyn GeoBBox>),
    /// A `GeoCircle`.
    Circle(Arc<dyn GeoCircle>),
    /// A `GeoPath`.
    Path(Arc<dyn GeoPath>),
    /// `GeoCompositeAreaShape`.
    AreaShape(Arc<dyn GeoAreaShape>),
    /// `GeoCompositeMembershipShape`.
    MembershipShape(Arc<dyn GeoMembershipShape>),
}

/// The simple name of a class code's class (Java's `getClass().getName()`
/// without the package).
fn cast_error(from: &StandardObject, to: &str) -> Error {
    Error::Runtime(format!(
        "Cannot cast {PACKAGE}.{} to {PACKAGE}.{to}",
        from.class_name()
    ))
}

impl StandardObject {
    /// The object's simple class name.
    pub fn class_name(&self) -> &'static str {
        let code = match self {
            StandardObject::Point(_) => Some(0),
            StandardObject::Solid(s) => s.class_code(),
            StandardObject::Polygon(s) => s.class_code(),
            StandardObject::PointShape(s) => s.class_code(),
            StandardObject::BBox(s) => s.class_code(),
            StandardObject::Circle(s) => s.class_code(),
            StandardObject::Path(s) => s.class_code(),
            StandardObject::AreaShape(s) => s.class_code(),
            StandardObject::MembershipShape(s) => s.class_code(),
        };
        code.and_then(class_name).unwrap_or("?")
    }

    /// The object as a `PlanetObject`, or `None` for a `GeoPoint`.
    pub fn as_planet_object(&self) -> Option<Arc<dyn PlanetObject>> {
        Some(match self {
            StandardObject::Point(_) => return None,
            StandardObject::Solid(s) => s.clone(),
            StandardObject::Polygon(s) => s.clone(),
            StandardObject::PointShape(s) => s.clone(),
            StandardObject::BBox(s) => s.clone(),
            StandardObject::Circle(s) => s.clone(),
            StandardObject::Path(s) => s.clone(),
            StandardObject::AreaShape(s) => s.clone(),
            StandardObject::MembershipShape(s) => s.clone(),
        })
    }

    /// `(GeoMembershipShape) object`.
    pub fn into_membership_shape(self) -> Result<Arc<dyn GeoMembershipShape>> {
        match self {
            StandardObject::MembershipShape(s) => Ok(s),
            other => other
                .into_area_shape()
                .map(|s| s as Arc<dyn GeoMembershipShape>),
        }
    }

    /// `(GeoAreaShape) object`.
    pub fn into_area_shape(self) -> Result<Arc<dyn GeoAreaShape>> {
        Ok(match self {
            StandardObject::Polygon(s) => s,
            StandardObject::PointShape(s) => s,
            StandardObject::BBox(s) => s,
            StandardObject::Circle(s) => s,
            StandardObject::Path(s) => s,
            StandardObject::AreaShape(s) => s,
            other => return Err(cast_error(&other, "GeoAreaShape")),
        })
    }

    /// `(GeoPolygon) object`.
    pub fn into_polygon(self) -> Result<Arc<dyn GeoPolygon>> {
        match self {
            StandardObject::Polygon(s) => Ok(s),
            other => Err(cast_error(&other, "GeoPolygon")),
        }
    }

    /// `(GeoPoint) object`.
    pub fn into_point(self) -> Result<GeoPoint> {
        match self {
            StandardObject::Point(p) => Ok(p),
            other => Err(cast_error(&other, "GeoPoint")),
        }
    }
}

/// `readClass(inputStream)`: the registry code of the class to read.
fn read_class(input: &mut Input<'_>) -> Result<u8> {
    if read_boolean(input)? {
        let index = input.read_byte();
        match u8::try_from(index)
            .ok()
            .filter(|&c| class_name(c).is_some())
        {
            Some(code) => Ok(code),
            None => Err(Error::Io(format!(
                "No standard object found for index: {index}"
            ))),
        }
    } else {
        // Java loads any `SerializableObject` by name. Every geo3d class
        // that can be constructed from a stream is registered, so a
        // registered class named in full resolves to its code; any other
        // name fails as a class Java could not find (an abstract or
        // package-private base class Java would find and then fail to
        // instantiate fails here too, with a different message).
        let name = read_string(input)?;
        let code = name
            .strip_prefix(PACKAGE)
            .and_then(|rest| rest.strip_prefix('.'))
            .and_then(|simple| CLASS_NAMES.iter().position(|&n| n == simple));
        match code {
            Some(code) => Ok(code as u8),
            None => Err(Error::Io(format!(
                "Can't find or access class of correct type for deserialization: {name}"
            ))),
        }
    }
}

/// `readObject(planetModel, inputStream)`: the class, then the object.
pub fn read_object(
    planet_model: &Arc<PlanetModel>,
    input: &mut Input<'_>,
) -> Result<StandardObject> {
    let code = read_class(input)?;
    input.nested(|input| read_object_of(planet_model, input, code))
}

/// `readObject(inputStream)`: an object whose class has an
/// `(InputStream)` constructor -- only `GeoPoint` and `PlanetModel`;
/// `PlanetModel` is not returned as a [`StandardObject`] and fails as a
/// class without the constructor would.
pub fn read_object_without_planet(input: &mut Input<'_>) -> Result<StandardObject> {
    let code = read_class(input)?;
    if code == 0 {
        return GeoPoint::read(input)
            .map(StandardObject::Point)
            .map_err(|_| instantiation_error(code));
    }
    let name = class_name(code).unwrap_or("?");
    Err(Error::Io(format!(
        "No such method exception for class {PACKAGE}.{name}: {PACKAGE}.{name}.<init>(java.io.InputStream)"
    )))
}

/// Java's reflective construction wraps whatever the constructor throws in
/// an `InvocationTargetException`, whose message is `null`.
fn instantiation_error(code: u8) -> Error {
    let name = class_name(code).unwrap_or("?");
    Error::Io(format!(
        "Exception instantiating class {PACKAGE}.{name}: null"
    ))
}

/// `readObject(planetModel, inputStream, clazz)` for a registry code.
pub fn read_object_of(
    planet_model: &Arc<PlanetModel>,
    input: &mut Input<'_>,
    code: u8,
) -> Result<StandardObject> {
    use super::*;
    let pm = planet_model;
    let built: Result<StandardObject> = match code {
        0 => GeoPoint::read(input).map(StandardObject::Point),
        1 => {
            geo_rectangle::GeoRectangle::read(pm, input).map(|s| StandardObject::BBox(Arc::new(s)))
        }
        2 => geo_standard_circle::GeoStandardCircle::read(pm, input)
            .map(|s| StandardObject::Circle(Arc::new(s))),
        3 => geo_standard_path::GeoStandardPath::read(pm, input)
            .map(|s| StandardObject::Path(Arc::new(s))),
        4 => geo_convex_polygon::GeoConvexPolygon::read(pm, input)
            .map(|s| StandardObject::Polygon(Arc::new(s))),
        5 => geo_convex_polygon::GeoConcavePolygon::read(pm, input)
            .map(|s| StandardObject::Polygon(Arc::new(s))),
        6 => geo_complex_polygon::GeoComplexPolygon::read(pm, input)
            .map(|s| StandardObject::Polygon(Arc::new(s))),
        7 => geo_composite::GeoCompositePolygon::read(pm, input)
            .map(|s| StandardObject::Polygon(Arc::new(s))),
        8 => geo_composite::GeoCompositeMembershipShape::read(pm, input)
            .map(|s| StandardObject::MembershipShape(Arc::new(s))),
        9 => geo_composite::GeoCompositeAreaShape::read(pm, input)
            .map(|s| StandardObject::AreaShape(Arc::new(s))),
        10 => geo_degenerate_point::GeoDegeneratePoint::read(pm, input)
            .map(|s| StandardObject::PointShape(Arc::new(s))),
        11 => geo_degenerate_horizontal_line::GeoDegenerateHorizontalLine::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        12 => geo_degenerate_vertical_line::GeoDegenerateLatitudeZone::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        13 => geo_degenerate_vertical_line::GeoDegenerateLongitudeSlice::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        14 => geo_degenerate_vertical_line::GeoDegenerateVerticalLine::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        15 => geo_latitude_zone::GeoLatitudeZone::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        16 => geo_longitude_slice::GeoLongitudeSlice::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        17 => geo_latitude_zone::GeoNorthLatitudeZone::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        18 => geo_north_rectangle::GeoNorthRectangle::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        19 => geo_latitude_zone::GeoSouthLatitudeZone::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        20 => geo_south_rectangle::GeoSouthRectangle::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        21 => geo_degenerate_horizontal_line::GeoWideDegenerateHorizontalLine::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        22 => geo_longitude_slice::GeoWideLongitudeSlice::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        23 => geo_wide_north_rectangle::GeoWideNorthRectangle::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        24 => geo_wide_rectangle::GeoWideRectangle::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        25 => geo_wide_south_rectangle::GeoWideSouthRectangle::read(pm, input)
            .map(|s| StandardObject::BBox(Arc::new(s))),
        26 => geo_world::GeoWorld::read(pm, input).map(|s| StandardObject::BBox(Arc::new(s))),
        27 => xyz_solid::DXdYdZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        28 => xyz_solid::DXdYZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        29 => xyz_solid::DXYdZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        30 => xyz_solid::DXYZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        31 => xyz_solid::XdYdZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        32 => xyz_solid::XdYZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        33 => xyz_solid::XYdZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s))),
        34 => {
            xyz_solid::StandardXYZSolid::read(pm, input).map(|s| StandardObject::Solid(Arc::new(s)))
        }
        36 => geo_degenerate_path::GeoDegeneratePath::read(pm, input)
            .map(|s| StandardObject::Path(Arc::new(s))),
        37 => geo_exact_circle::GeoExactCircle::read(pm, input)
            .map(|s| StandardObject::Circle(Arc::new(s))),
        38 => {
            geo_s2_shape::GeoS2Shape::read(pm, input).map(|s| StandardObject::Polygon(Arc::new(s)))
        }
        // 35 is `PlanetModel`, which has no `(PlanetModel, InputStream)`
        // constructor; `read_class` admits no other code.
        _ => Err(Error::Runtime(String::new())),
    };
    match built {
        Ok(object) => Ok(object),
        Err(_) if code == 35 => Err(Error::Io(format!(
            "No such method exception for class {PACKAGE}.PlanetModel: \
             {PACKAGE}.PlanetModel.<init>({PACKAGE}.PlanetModel,java.io.InputStream)"
        ))),
        Err(_) => Err(instantiation_error(code)),
    }
}

/// `readPlanetObject(inputStream)`: a planet model, then an object on it.
pub fn read_planet_object(input: &mut Input<'_>) -> Result<StandardObject> {
    let planet_model = Arc::new(PlanetModel::read(input)?);
    let object = read_object(&planet_model, input)?;
    if object.as_planet_object().is_none() {
        return Err(Error::Io(format!(
            "Type of object is not expected PlanetObject: {PACKAGE}.{}",
            object.class_name()
        )));
    }
    Ok(object)
}

/// `writeHeterogeneousArray(outputStream, values)`: the count, then each
/// object with its class.
pub fn write_heterogeneous_array<T: SerializableObject + ?Sized>(
    out: &mut Vec<u8>,
    values: &[Arc<T>],
) -> Result<()> {
    write_count(out, values.len());
    for v in values {
        write_object(out, &**v)?;
    }
    Ok(())
}

/// `readHeterogeneousArray(planetModel, inputStream, clazz)` before the
/// casts.
pub fn read_heterogeneous_array(
    planet_model: &Arc<PlanetModel>,
    input: &mut Input<'_>,
) -> Result<Vec<StandardObject>> {
    let count = super::serializable::read_count(input)?;
    // Grown as read: a count off the stream must not size an allocation.
    let mut rval = Vec::new();
    for _ in 0..count {
        rval.push(read_object(planet_model, input)?);
    }
    Ok(rval)
}

/// `writePolygonArray(outputStream, values)`.
pub fn write_polygon_array(out: &mut Vec<u8>, values: &[Arc<dyn GeoPolygon>]) -> Result<()> {
    write_heterogeneous_array(out, values)
}

/// `readPolygonArray(planetModel, inputStream)`.
pub fn read_polygon_array(
    planet_model: &Arc<PlanetModel>,
    input: &mut Input<'_>,
) -> Result<Vec<Arc<dyn GeoPolygon>>> {
    read_heterogeneous_array(planet_model, input)?
        .into_iter()
        .map(StandardObject::into_polygon)
        .collect()
}
