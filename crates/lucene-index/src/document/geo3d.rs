//! The spatial3d fields: `Geo3DPoint` and `Geo3DDocValuesField`
//! (`org.apache.lucene.spatial3d`).
//!
//! What reaches disk:
//!
//! - `Geo3DPoint`: a three-dimension, four-byte point -- the surface point's
//!   x, y and z, each `PlanetModel.encodeValue`d to an `int` and written
//!   `NumericUtils.intToSortableBytes`.
//! - `Geo3DDocValuesField`: one `SORTED_NUMERIC` value per point, the x/y/z
//!   packed into a `long` by the planet model's `DocValueEncoder` (21 bits
//!   each).
//!
//! The query (`PointInGeo3DShapeQuery`) and the distance sorts live in
//! `lucene_search::document::geo`, which reads what these write.

use std::borrow::Cow;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_util::geo::java_double_string;
use lucene_util::geo::GeoUtils;
use lucene_util::spatial3d::{GeoPoint, PlanetModel};

use super::numeric::{int_to_sortable_bytes, sortable_bytes_to_int};
use super::{illegal, DocValuesType, FieldTokens, FieldType, IndexableField, Number, Result};

fn s3d(e: lucene_util::spatial3d::Error) -> super::Error {
    illegal(e.to_string())
}

/// `Geo3DUtil.RADIANS_PER_DEGREE`.
pub const RADIANS_PER_DEGREE: f64 = std::f64::consts::PI / 180.0;

/// `Geo3DUtil.fromDegrees(degrees)`: `degrees * RADIANS_PER_DEGREE` (not
/// `Math.toRadians`, whose constant rounds differently).
pub fn from_degrees(degrees: f64) -> f64 {
    degrees * RADIANS_PER_DEGREE
}

/// `Geo3DPoint`: an indexed point on a planet, as x/y/z.
#[derive(Debug, Clone)]
pub struct Geo3DPoint {
    name: String,
    field_type: FieldType,
    planet_model: Arc<PlanetModel>,
    packed: [u8; 12],
}

impl Geo3DPoint {
    /// `Geo3DPoint.TYPE`: three dimensions of four bytes.
    pub fn field_type_of() -> FieldType {
        let mut ft = FieldType::new();
        ft.set_dimensions(3, 4)
            .expect("three four-byte dimensions are valid");
        ft.frozen()
    }

    /// `Geo3DPoint(name, latitude, longitude)`: on WGS84, latitude and
    /// longitude in degrees.
    ///
    /// # Errors
    /// An invalid latitude or longitude, with `GeoUtils`' message.
    pub fn new(name: impl Into<String>, latitude: f64, longitude: f64) -> Result<Self> {
        Self::with_planet_model(name, &PlanetModel::wgs84(), latitude, longitude)
    }

    /// `Geo3DPoint(name, planetModel, latitude, longitude)`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn with_planet_model(
        name: impl Into<String>,
        planet_model: &Arc<PlanetModel>,
        latitude: f64,
        longitude: f64,
    ) -> Result<Self> {
        GeoUtils::check_latitude(latitude).map_err(|e| illegal(e.to_string()))?;
        GeoUtils::check_longitude(longitude).map_err(|e| illegal(e.to_string()))?;
        let point = GeoPoint::from_lat_lon(
            planet_model,
            from_degrees(latitude),
            from_degrees(longitude),
        )
        .map_err(s3d)?;
        Self::from_xyz_on(name, planet_model, point.x, point.y, point.z)
    }

    /// `Geo3DPoint(name, x, y, z)`: on WGS84.
    ///
    /// # Errors
    /// A coordinate outside the planet model's encodable range.
    pub fn from_xyz(name: impl Into<String>, x: f64, y: f64, z: f64) -> Result<Self> {
        Self::from_xyz_on(name, &PlanetModel::wgs84(), x, y, z)
    }

    /// `Geo3DPoint(name, planetModel, x, y, z)`.
    ///
    /// # Errors
    /// As [`Self::from_xyz`].
    pub fn from_xyz_on(
        name: impl Into<String>,
        planet_model: &Arc<PlanetModel>,
        x: f64,
        y: f64,
        z: f64,
    ) -> Result<Self> {
        let mut packed = [0u8; 12];
        Self::encode_dimension(x, &mut packed[..4], planet_model)?;
        Self::encode_dimension(y, &mut packed[4..8], planet_model)?;
        Self::encode_dimension(z, &mut packed[8..], planet_model)?;
        Ok(Geo3DPoint {
            name: name.into(),
            field_type: Self::field_type_of(),
            planet_model: planet_model.clone(),
            packed,
        })
    }

    /// `encodeDimension(value, bytes, offset, planetModel)`: four sortable
    /// bytes of `planetModel.encodeValue(value)` into `out[..4]`.
    ///
    /// # Errors
    /// A value outside the planet model's range, with Java's message.
    pub fn encode_dimension(value: f64, out: &mut [u8], planet_model: &PlanetModel) -> Result<()> {
        let encoded = planet_model.encode_value(value).map_err(s3d)?;
        out[..4].copy_from_slice(&int_to_sortable_bytes(encoded));
        Ok(())
    }

    /// `decodeDimension(value, offset, planetModel)`: of `bytes[..4]`.
    pub fn decode_dimension(bytes: &[u8], planet_model: &PlanetModel) -> f64 {
        planet_model.decode_value(sortable_bytes_to_int(bytes))
    }

    /// The packed point.
    pub fn packed(&self) -> &[u8; 12] {
        &self.packed
    }

    /// The decoded x, y and z.
    pub fn xyz(&self) -> (f64, f64, f64) {
        let pm = &*self.planet_model;
        (
            Self::decode_dimension(&self.packed[..4], pm),
            Self::decode_dimension(&self.packed[4..8], pm),
            Self::decode_dimension(&self.packed[8..], pm),
        )
    }
}

impl std::fmt::Display for Geo3DPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (x, y, z) = self.xyz();
        write!(
            f,
            "Geo3DPoint <{}: x={} y={} z={}>",
            self.name,
            java_double_string(x),
            java_double_string(y),
            java_double_string(z)
        )
    }
}

impl IndexableField for Geo3DPoint {
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

/// `Geo3DDocValuesField`: a point as a `SORTED_NUMERIC` doc value, for the
/// distance sorts.
#[derive(Debug, Clone)]
pub struct Geo3DDocValuesField {
    name: String,
    field_type: FieldType,
    planet_model: Arc<PlanetModel>,
    value: i64,
}

impl Geo3DDocValuesField {
    /// `Geo3DDocValuesField.TYPE`.
    pub fn field_type_of() -> FieldType {
        let mut ft = FieldType::new();
        ft.set_doc_values_type(DocValuesType::SortedNumeric)
            .expect("unfrozen");
        ft.frozen()
    }

    /// `Geo3DDocValuesField(name, point, planetModel)`.
    ///
    /// # Errors
    /// A coordinate the encoder cannot hold, with Java's message.
    pub fn new(
        name: impl Into<String>,
        point: &GeoPoint,
        planet_model: &Arc<PlanetModel>,
    ) -> Result<Self> {
        Self::from_xyz(name, point.x, point.y, point.z, planet_model)
    }

    /// `Geo3DDocValuesField(name, x, y, z, planetModel)`.
    ///
    /// # Errors
    /// As [`Self::new`].
    pub fn from_xyz(
        name: impl Into<String>,
        x: f64,
        y: f64,
        z: f64,
        planet_model: &Arc<PlanetModel>,
    ) -> Result<Self> {
        Ok(Geo3DDocValuesField {
            name: name.into(),
            field_type: Self::field_type_of(),
            planet_model: planet_model.clone(),
            value: planet_model
                .doc_value_encoder()
                .encode_point_xyz(x, y, z)
                .map_err(s3d)?,
        })
    }

    /// `setLocationValue(x, y, z)`.
    ///
    /// # Errors
    /// As [`Self::new`]; the value is unchanged then.
    pub fn set_location_value(&mut self, x: f64, y: f64, z: f64) -> Result<()> {
        self.value = self
            .planet_model
            .doc_value_encoder()
            .encode_point_xyz(x, y, z)
            .map_err(s3d)?;
        Ok(())
    }

    /// The encoded doc value.
    pub fn value(&self) -> i64 {
        self.value
    }
}

impl std::fmt::Display for Geo3DDocValuesField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let e = self.planet_model.doc_value_encoder();
        write!(
            f,
            "Geo3DDocValuesField <{}:{},{},{}>",
            self.name,
            java_double_string(e.decode_x_value(self.value)),
            java_double_string(e.decode_y_value(self.value)),
            java_double_string(e.decode_z_value(self.value))
        )
    }
}

impl IndexableField for Geo3DDocValuesField {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_packs_three_encoded_dimensions() {
        let pm = PlanetModel::wgs84();
        let p = Geo3DPoint::new("f", 40.5, -73.25).unwrap();
        let g = GeoPoint::from_lat_lon(&pm, from_degrees(40.5), from_degrees(-73.25)).unwrap();
        assert_eq!(
            &p.packed()[..4],
            &int_to_sortable_bytes(pm.encode_value(g.x).unwrap())
        );
        assert_eq!(
            &p.packed()[8..],
            &int_to_sortable_bytes(pm.encode_value(g.z).unwrap())
        );
        let (x, _, _) = p.xyz();
        assert_eq!(x, pm.decode_value(pm.encode_value(g.x).unwrap()));
        assert!(p.to_string().starts_with("Geo3DPoint <f: x="));
        assert_eq!(p.name(), "f");
        assert_eq!(p.field_type().point_dimension_count(), 3);
        assert!(p.binary_value().is_some());
        assert!(p.token_stream(&Analyzer::standard(None)).unwrap().is_none());
        assert!(Geo3DPoint::new("f", 91.0, 0.0).is_err());
        assert!(Geo3DPoint::new("f", 0.0, 181.0).is_err());
        assert!(Geo3DPoint::from_xyz("f", 2.0, 0.0, 0.0).is_err());
        assert!(Geo3DPoint::from_xyz("f", 0.5, 0.5, 0.5).is_ok());
    }

    #[test]
    fn doc_value_packs_the_encoder_value() {
        let pm = PlanetModel::sphere();
        let g = GeoPoint::from_lat_lon(&pm, 0.3, 0.4).unwrap();
        let mut f = Geo3DDocValuesField::new("d", &g, &pm).unwrap();
        assert_eq!(f.value(), pm.doc_value_encoder().encode_point(&g).unwrap());
        assert!(f.to_string().starts_with("Geo3DDocValuesField <d:"));
        assert!(f.set_location_value(5.0, 0.0, 0.0).is_err());
        f.set_location_value(1.0, 0.0, 0.0).unwrap();
        assert_eq!(f.numeric_value(), Some(Number::Long(f.value())));
        assert_eq!(f.string_value().unwrap(), f.value().to_string());
        assert_eq!(f.name(), "d");
        assert_eq!(
            f.field_type().doc_values_type(),
            DocValuesType::SortedNumeric
        );
        assert!(f.token_stream(&Analyzer::standard(None)).unwrap().is_none());
    }
}
