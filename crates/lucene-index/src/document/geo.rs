//! The geo point fields: `LatLonPoint`, `LatLonDocValuesField`,
//! `XYPointField` and `XYDocValuesField`.
//!
//! What reaches disk:
//!
//! - `LatLonPoint`: a two-dimension, four-byte point -- latitude then
//!   longitude, each `GeoEncodingUtils.encodeLatitude`/`encodeLongitude`d to
//!   an `int` and written `NumericUtils.intToSortableBytes` (big-endian, sign
//!   bit flipped).
//! - `XYPointField`: the same shape over `XYEncodingUtils.encode` of `x`
//!   then `y`.
//! - `LatLonDocValuesField`/`XYDocValuesField`: one `SORTED_NUMERIC` value
//!   per point, the two encoded `int`s packed into a `long` -- latitude (`x`)
//!   in the high half, longitude (`y`) in the low half.
//!
//! The query factories, distance sorts and `LatLonPoint.nearest` live in
//! `lucene_search::document::geo`, which reads what these write.

use std::borrow::Cow;

use lucene_analysis::Analyzer;
use lucene_util::geo::{GeoEncodingUtils, GeoError, XYEncodingUtils};

use super::numeric::int_to_sortable_bytes;
use super::{illegal, DocValuesType, FieldTokens, FieldType, IndexableField, Number, Result};

fn geo(e: GeoError) -> super::Error {
    illegal(e.to_string())
}

/// `LatLonPoint.TYPE` / `XYPointField.TYPE`: two dimensions of four bytes.
fn point_type() -> FieldType {
    let mut ft = FieldType::new();
    ft.set_dimensions(2, 4)
        .expect("two four-byte dimensions are valid");
    ft.frozen()
}

/// `LatLonDocValuesField.TYPE` / `XYDocValuesField.TYPE`.
fn doc_values_type() -> FieldType {
    let mut ft = FieldType::new();
    ft.set_doc_values_type(DocValuesType::SortedNumeric)
        .expect("unfrozen");
    ft.frozen()
}

/// Two encoded `int`s as the eight packed point bytes.
fn pack(a: i32, b: i32) -> [u8; 8] {
    let mut out = [0u8; 8];
    out[..4].copy_from_slice(&int_to_sortable_bytes(a));
    out[4..].copy_from_slice(&int_to_sortable_bytes(b));
    out
}

/// Two encoded `int`s as one doc value: `(long) a << 32 | b & 0xFFFFFFFFL`.
pub fn pack_doc_value(a: i32, b: i32) -> i64 {
    (i64::from(a) << 32) | i64::from(b as u32)
}

/// The high `int` of a packed doc value (`(int) (value >> 32)`).
pub fn doc_value_high(value: i64) -> i32 {
    (value >> 32) as i32
}

/// The low `int` of a packed doc value (`(int) (value & 0xFFFFFFFF)`).
pub fn doc_value_low(value: i64) -> i32 {
    value as i32
}

/// `LatLonPoint`: an indexed latitude/longitude point.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonPoint {
    name: String,
    field_type: FieldType,
    packed: [u8; 8],
}

impl LatLonPoint {
    /// `BYTES`: the width of one dimension.
    pub const BYTES: usize = 4;

    /// `new LatLonPoint(name, latitude, longitude)`.
    ///
    /// # Errors
    /// An invalid latitude or longitude, with `GeoUtils`' message.
    pub fn new(name: impl Into<String>, latitude: f64, longitude: f64) -> Result<Self> {
        Ok(LatLonPoint {
            name: name.into(),
            field_type: point_type(),
            packed: Self::encode(latitude, longitude)?,
        })
    }

    /// `LatLonPoint.TYPE`.
    pub fn field_type_of() -> FieldType {
        point_type()
    }

    /// `setLocationValue(latitude, longitude)`.
    ///
    /// # Errors
    /// As [`Self::new`]; the value is unchanged then.
    pub fn set_location_value(&mut self, latitude: f64, longitude: f64) -> Result<()> {
        self.packed = Self::encode(latitude, longitude)?;
        Ok(())
    }

    /// The private `encode(latitude, longitude)`: the packed point.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn encode(latitude: f64, longitude: f64) -> Result<[u8; 8]> {
        Ok(pack(
            GeoEncodingUtils::encode_latitude(latitude).map_err(geo)?,
            GeoEncodingUtils::encode_longitude(longitude).map_err(geo)?,
        ))
    }

    /// The private `encodeCeil(latitude, longitude)`: the packed point,
    /// rounded up.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn encode_ceil(latitude: f64, longitude: f64) -> Result<[u8; 8]> {
        Ok(pack(
            GeoEncodingUtils::encode_latitude_ceil(latitude).map_err(geo)?,
            GeoEncodingUtils::encode_longitude_ceil(longitude).map_err(geo)?,
        ))
    }

    /// The packed point.
    pub fn packed(&self) -> &[u8; 8] {
        &self.packed
    }

    /// The decoded latitude and longitude.
    pub fn location(&self) -> (f64, f64) {
        (
            GeoEncodingUtils::decode_latitude_bytes(&self.packed, 0),
            GeoEncodingUtils::decode_longitude_bytes(&self.packed, 4),
        )
    }
}

impl std::fmt::Display for LatLonPoint {
    /// `toString()`: `LatLonPoint <name:lat,lon>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (lat, lon) = self.location();
        write!(
            f,
            "LatLonPoint <{}:{},{}>",
            self.name,
            lucene_util::geo::java_double_string(lat),
            lucene_util::geo::java_double_string(lon)
        )
    }
}

/// `XYPointField`: an indexed cartesian point.
#[derive(Debug, Clone, PartialEq)]
pub struct XYPointField {
    name: String,
    field_type: FieldType,
    packed: [u8; 8],
}

impl XYPointField {
    /// `BYTES`: the width of one dimension.
    pub const BYTES: usize = 4;

    /// `new XYPointField(name, x, y)`.
    ///
    /// # Errors
    /// A non-finite coordinate, with `XYEncodingUtils`' message.
    pub fn new(name: impl Into<String>, x: f32, y: f32) -> Result<Self> {
        Ok(XYPointField {
            name: name.into(),
            field_type: point_type(),
            packed: Self::encode(x, y)?,
        })
    }

    /// `XYPointField.TYPE`.
    pub fn field_type_of() -> FieldType {
        point_type()
    }

    /// `setLocationValue(x, y)`.
    ///
    /// # Errors
    /// As [`Self::new`]; the value is unchanged then.
    pub fn set_location_value(&mut self, x: f32, y: f32) -> Result<()> {
        self.packed = Self::encode(x, y)?;
        Ok(())
    }

    /// The packed point of `(x, y)`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn encode(x: f32, y: f32) -> Result<[u8; 8]> {
        Ok(pack(
            XYEncodingUtils::encode(x).map_err(geo)?,
            XYEncodingUtils::encode(y).map_err(geo)?,
        ))
    }

    /// The packed point.
    pub fn packed(&self) -> &[u8; 8] {
        &self.packed
    }

    /// The decoded `x` and `y`.
    pub fn location(&self) -> (f32, f32) {
        (
            XYEncodingUtils::decode_bytes(&self.packed, 0),
            XYEncodingUtils::decode_bytes(&self.packed, 4),
        )
    }
}

impl std::fmt::Display for XYPointField {
    /// `toString()`: `XYPointField <name:x,y>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y) = self.location();
        write!(
            f,
            "XYPointField <{}:{},{}>",
            self.name,
            lucene_util::geo::java_float_string(x),
            lucene_util::geo::java_float_string(y)
        )
    }
}

macro_rules! point_indexable {
    ($ty:ident) => {
        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                Some(Cow::Borrowed(&self.packed))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

point_indexable!(LatLonPoint);
point_indexable!(XYPointField);

/// `LatLonDocValuesField`: a latitude/longitude point as a `SORTED_NUMERIC`
/// doc value, for sorting by distance and the slow doc-values queries.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonDocValuesField {
    name: String,
    field_type: FieldType,
    value: i64,
}

impl LatLonDocValuesField {
    /// `new LatLonDocValuesField(name, latitude, longitude)`.
    ///
    /// # Errors
    /// An invalid latitude or longitude, with `GeoUtils`' message.
    pub fn new(name: impl Into<String>, latitude: f64, longitude: f64) -> Result<Self> {
        Ok(LatLonDocValuesField {
            name: name.into(),
            field_type: doc_values_type(),
            value: Self::encode(latitude, longitude)?,
        })
    }

    /// `LatLonDocValuesField.TYPE`.
    pub fn field_type_of() -> FieldType {
        doc_values_type()
    }

    /// `setLocationValue(latitude, longitude)`.
    ///
    /// # Errors
    /// As [`Self::new`]; the value is unchanged then.
    pub fn set_location_value(&mut self, latitude: f64, longitude: f64) -> Result<()> {
        self.value = Self::encode(latitude, longitude)?;
        Ok(())
    }

    /// The doc value of a point.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn encode(latitude: f64, longitude: f64) -> Result<i64> {
        Ok(pack_doc_value(
            GeoEncodingUtils::encode_latitude(latitude).map_err(geo)?,
            GeoEncodingUtils::encode_longitude(longitude).map_err(geo)?,
        ))
    }

    /// The doc value.
    pub fn value(&self) -> i64 {
        self.value
    }
}

impl std::fmt::Display for LatLonDocValuesField {
    /// `toString()`: `LatLonDocValuesField <name:lat,lon>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "LatLonDocValuesField <{}:{},{}>",
            self.name,
            lucene_util::geo::java_double_string(GeoEncodingUtils::decode_latitude(
                doc_value_high(self.value)
            )),
            lucene_util::geo::java_double_string(GeoEncodingUtils::decode_longitude(
                doc_value_low(self.value)
            ))
        )
    }
}

/// `XYDocValuesField`: a cartesian point as a `SORTED_NUMERIC` doc value.
#[derive(Debug, Clone, PartialEq)]
pub struct XYDocValuesField {
    name: String,
    field_type: FieldType,
    value: i64,
}

impl XYDocValuesField {
    /// `new XYDocValuesField(name, x, y)`.
    ///
    /// # Errors
    /// A non-finite coordinate, with `XYEncodingUtils`' message.
    pub fn new(name: impl Into<String>, x: f32, y: f32) -> Result<Self> {
        Ok(XYDocValuesField {
            name: name.into(),
            field_type: doc_values_type(),
            value: Self::encode(x, y)?,
        })
    }

    /// `XYDocValuesField.TYPE`.
    pub fn field_type_of() -> FieldType {
        doc_values_type()
    }

    /// `setLocationValue(x, y)`.
    ///
    /// # Errors
    /// As [`Self::new`]; the value is unchanged then.
    pub fn set_location_value(&mut self, x: f32, y: f32) -> Result<()> {
        self.value = Self::encode(x, y)?;
        Ok(())
    }

    /// The doc value of a point.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn encode(x: f32, y: f32) -> Result<i64> {
        Ok(pack_doc_value(
            XYEncodingUtils::encode(x).map_err(geo)?,
            XYEncodingUtils::encode(y).map_err(geo)?,
        ))
    }

    /// The doc value.
    pub fn value(&self) -> i64 {
        self.value
    }
}

impl std::fmt::Display for XYDocValuesField {
    /// `toString()`: `XYDocValuesField <name:x,y>`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "XYDocValuesField <{}:{},{}>",
            self.name,
            lucene_util::geo::java_float_string(XYEncodingUtils::decode(doc_value_high(
                self.value
            ))),
            lucene_util::geo::java_float_string(XYEncodingUtils::decode(doc_value_low(self.value)))
        )
    }
}

macro_rules! dv_indexable {
    ($ty:ident) => {
        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            fn numeric_value(&self) -> Option<Number> {
                Some(Number::Long(self.value))
            }
            fn string_value(&self) -> Option<Cow<'_, str>> {
                Some(Cow::Owned(self.value.to_string()))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

dv_indexable!(LatLonDocValuesField);
dv_indexable!(XYDocValuesField);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lat_lon_point_packs_both_encodings() {
        let p = LatLonPoint::new("f", 40.5, -73.25).unwrap();
        let lat = GeoEncodingUtils::encode_latitude(40.5).unwrap();
        let lon = GeoEncodingUtils::encode_longitude(-73.25).unwrap();
        assert_eq!(&p.packed()[..4], &int_to_sortable_bytes(lat));
        assert_eq!(&p.packed()[4..], &int_to_sortable_bytes(lon));
        assert_eq!(p.binary_value().unwrap().as_ref(), p.packed());
        assert_eq!(p.name(), "f");
        assert_eq!(p.field_type().point_dimension_count(), 2);
        assert_eq!(p.field_type().point_num_bytes(), 4);
        assert!(p.numeric_value().is_none());
        assert!(p.token_stream(&Analyzer::standard(None)).unwrap().is_none());
        let (la, lo) = p.location();
        assert!((la - 40.5).abs() < 1e-6 && (lo + 73.25).abs() < 1e-6);
        assert!(p.to_string().starts_with("LatLonPoint <f:40.4999"));
        let ceil = LatLonPoint::encode_ceil(40.5, -73.25).unwrap();
        assert!(ceil >= *p.packed());
        assert_eq!(LatLonPoint::field_type_of(), *p.field_type());
    }

    #[test]
    fn invalid_coordinates_are_rejected_with_java_messages() {
        let e = LatLonPoint::new("f", 91.0, 0.0).unwrap_err();
        assert!(e.to_string().contains("invalid latitude 91.0"), "{e}");
        let e = LatLonDocValuesField::new("f", 0.0, -181.0).unwrap_err();
        assert!(e.to_string().contains("invalid longitude -181.0"), "{e}");
        assert!(XYPointField::new("f", f32::NAN, 0.0).is_err());
        assert!(XYDocValuesField::new("f", 0.0, f32::INFINITY).is_err());
        let mut p = LatLonPoint::new("f", 1.0, 2.0).unwrap();
        let before = *p.packed();
        assert!(p.set_location_value(100.0, 0.0).is_err());
        assert_eq!(*p.packed(), before);
        p.set_location_value(-90.0, 180.0).unwrap();
        assert_eq!(p.location().0, -90.0);
        assert!(LatLonPoint::encode_ceil(f64::NAN, 0.0).is_err());
    }

    #[test]
    fn doc_values_pack_high_and_low_halves() {
        let f = LatLonDocValuesField::new("g", -45.0, 170.0).unwrap();
        let lat = GeoEncodingUtils::encode_latitude(-45.0).unwrap();
        let lon = GeoEncodingUtils::encode_longitude(170.0).unwrap();
        assert_eq!(doc_value_high(f.value()), lat);
        assert_eq!(doc_value_low(f.value()), lon);
        assert_eq!(f.numeric_value(), Some(Number::Long(f.value())));
        assert_eq!(f.string_value().unwrap(), f.value().to_string());
        assert_eq!(
            f.field_type().doc_values_type(),
            DocValuesType::SortedNumeric
        );
        assert!(f.token_stream(&Analyzer::standard(None)).unwrap().is_none());
        assert_eq!(f.name(), "g");
        assert!(f.to_string().starts_with("LatLonDocValuesField <g:-45.0"));
        let mut f = f;
        f.set_location_value(1.0, 1.0).unwrap();
        assert_eq!(f.value(), LatLonDocValuesField::encode(1.0, 1.0).unwrap());
        assert_eq!(LatLonDocValuesField::field_type_of(), *f.field_type());
        assert_eq!(pack_doc_value(-1, -1), -1);
        assert_eq!(pack_doc_value(0, -1), 0xFFFF_FFFF);
    }

    #[test]
    fn xy_fields_encode_x_then_y() {
        let p = XYPointField::new("xy", 1.5, -2.25).unwrap();
        assert_eq!(p.location(), (1.5, -2.25));
        assert_eq!(p.to_string(), "XYPointField <xy:1.5,-2.25>");
        assert_eq!(p.binary_value().unwrap().len(), 8);
        assert_eq!(p.name(), "xy");
        assert!(p.token_stream(&Analyzer::standard(None)).unwrap().is_none());
        let mut p = p;
        p.set_location_value(3.0, 4.0).unwrap();
        assert_eq!(p.location(), (3.0, 4.0));
        assert_eq!(XYPointField::field_type_of(), *p.field_type());
        let d = XYDocValuesField::new("xy", 1.5, -2.25).unwrap();
        assert_eq!(XYEncodingUtils::decode(doc_value_high(d.value())), 1.5);
        assert_eq!(XYEncodingUtils::decode(doc_value_low(d.value())), -2.25);
        assert_eq!(d.to_string(), "XYDocValuesField <xy:1.5,-2.25>");
        assert_eq!(d.numeric_value(), Some(Number::Long(d.value())));
        assert!(d.string_value().is_some());
        assert_eq!(d.name(), "xy");
        assert!(d.token_stream(&Analyzer::standard(None)).unwrap().is_none());
        let mut d = d;
        d.set_location_value(0.0, 0.0).unwrap();
        assert_eq!(d.value(), XYDocValuesField::encode(0.0, 0.0).unwrap());
        assert_eq!(XYDocValuesField::field_type_of(), *d.field_type());
    }
}
