//! `PlanetModel` (`org.apache.lucene.spatial3d.geom.PlanetModel`): the
//! ellipsoid every geo3d computation is made on, the 32-bit-per-dimension
//! point encoding `Geo3DPoint` indexes, and the 21-bit-per-dimension
//! `DocValueEncoder` `Geo3DDocValuesField` stores.
//!
//! The surface is `x^2/a^2 + y^2/a^2 + z^2/c^2 = 1` with the radii scaled by
//! the mean radius `(2a + b) / 3`, so every distance is in planet radii (or
//! radians on the sphere).

use std::sync::{Arc, OnceLock};

use super::geo_point::GeoPoint;
use super::jmath::{abs, atan, atan2, cos, floor, max, min, next_down, rem, sin, sqrt, tan};
use super::serializable::{read_double, write_double, Input};
use super::vector::{Vector, MINIMUM_RESOLUTION};
use super::{Error, Result};

const BITS: u32 = 32;

/// A planet model.
#[derive(Debug, Clone)]
pub struct PlanetModel {
    /// Semi-major axis.
    pub a: f64,
    /// Semi-minor axis.
    pub b: f64,
    /// `xyScaling`.
    pub xy_scaling: f64,
    /// `zScaling`.
    pub z_scaling: f64,
    /// `inverseXYScaling`.
    pub inverse_xy_scaling: f64,
    /// `inverseZScaling`.
    pub inverse_z_scaling: f64,
    /// `inverseXYScalingSquared`.
    pub inverse_xy_scaling_squared: f64,
    /// `inverseZScalingSquared`.
    pub inverse_z_scaling_squared: f64,
    /// `scaledFlattening`.
    pub scaled_flattening: f64,
    /// `squareRatio`.
    pub square_ratio: f64,
    /// `meanRadius`, `(2a + b) / 3`.
    pub mean_radius: f64,
    /// `scale`.
    pub scale: f64,
    /// `inverseScale`.
    pub inverse_scale: f64,
    /// `NORTH_POLE`.
    pub north_pole: GeoPoint,
    /// `SOUTH_POLE`.
    pub south_pole: GeoPoint,
    /// `MIN_X_POLE`.
    pub min_x_pole: GeoPoint,
    /// `MAX_X_POLE`.
    pub max_x_pole: GeoPoint,
    /// `MIN_Y_POLE`.
    pub min_y_pole: GeoPoint,
    /// `MAX_Y_POLE`.
    pub max_y_pole: GeoPoint,
    /// `minimumPoleDistance`.
    pub minimum_pole_distance: f64,
    /// `MAX_VALUE`: the largest encodable magnitude.
    pub max_value: f64,
    mul: f64,
    /// `DECODE`: the width of one encoded step.
    pub decode: f64,
    /// `MAX_ENCODED_VALUE`.
    pub max_encoded_value: i32,
    /// `MIN_ENCODED_VALUE`.
    pub min_encoded_value: i32,
    doc_value_encoder: DocValueEncoder,
}

/// `equals`: the same axes.
impl PartialEq for PlanetModel {
    fn eq(&self, other: &PlanetModel) -> bool {
        self.a == other.a && self.b == other.b
    }
}

impl PlanetModel {
    /// `PlanetModel(semiMajorAxis, semiMinorAxis)`.
    pub fn new(semi_major_axis: f64, semi_minor_axis: f64) -> PlanetModel {
        let a = semi_major_axis;
        let b = semi_minor_axis;
        let mean_radius = (2.0 * semi_major_axis + semi_minor_axis) / 3.0;
        let xy_scaling = semi_major_axis / mean_radius;
        let z_scaling = semi_minor_axis / mean_radius;
        let scale = (2.0 * xy_scaling + z_scaling) / 3.0;
        let inverse_xy_scaling = 1.0 / xy_scaling;
        let inverse_z_scaling = 1.0 / z_scaling;
        let scaled_flattening = (xy_scaling - z_scaling) * inverse_xy_scaling;
        let square_ratio =
            (xy_scaling * xy_scaling - z_scaling * z_scaling) / (z_scaling * z_scaling);
        let inverse_xy_scaling_squared = inverse_xy_scaling * inverse_xy_scaling;
        let inverse_z_scaling_squared = inverse_z_scaling * inverse_z_scaling;
        use std::f64::consts::PI;
        // The poles carry in-range latitudes/longitudes, so these never fail.
        let pole = |m: f64, x: f64, y: f64, z: f64, lat: f64, lon: f64| {
            GeoPoint::with_magnitude_lat_lon(m, x, y, z, lat, lon).expect("pole in range")
        };
        let north_pole = pole(z_scaling, 0.0, 0.0, 1.0, PI * 0.5, 0.0);
        let south_pole = pole(z_scaling, 0.0, 0.0, -1.0, -PI * 0.5, 0.0);
        let min_x_pole = pole(xy_scaling, -1.0, 0.0, 0.0, 0.0, -PI);
        let max_x_pole = pole(xy_scaling, 1.0, 0.0, 0.0, 0.0, 0.0);
        let min_y_pole = pole(xy_scaling, 0.0, -1.0, 0.0, 0.0, -PI * 0.5);
        let max_y_pole = pole(xy_scaling, 0.0, 1.0, 0.0, 0.0, PI * 0.5);
        let inverse_scale = 1.0 / scale;
        let mut pm = PlanetModel {
            a,
            b,
            xy_scaling,
            z_scaling,
            inverse_xy_scaling,
            inverse_z_scaling,
            inverse_xy_scaling_squared,
            inverse_z_scaling_squared,
            scaled_flattening,
            square_ratio,
            mean_radius,
            scale,
            inverse_scale,
            north_pole,
            south_pole,
            min_x_pole,
            max_x_pole,
            min_y_pole,
            max_y_pole,
            minimum_pole_distance: 0.0,
            max_value: 0.0,
            mul: 0.0,
            decode: 0.0,
            max_encoded_value: 0,
            min_encoded_value: 0,
            doc_value_encoder: DocValueEncoder::default(),
        };
        pm.minimum_pole_distance = min(
            pm.surface_distance(&pm.north_pole, &pm.south_pole),
            pm.surface_distance(&pm.min_x_pole, &pm.max_x_pole),
        );
        pm.max_value = pm.maximum_magnitude();
        pm.mul = ((1u64 << BITS) as f64) / (2.0 * pm.max_value);
        pm.decode = next_safe_double(1.0 / pm.mul);
        // Both are in range by construction (`-MAX_VALUE`, `MAX_VALUE`).
        pm.min_encoded_value = pm.encode_value(-pm.max_value).unwrap_or(i32::MIN);
        pm.max_encoded_value = pm.encode_value(pm.max_value).unwrap_or(i32::MAX);
        pm.doc_value_encoder = DocValueEncoder::new(&pm);
        pm
    }

    /// `PlanetModel.SPHERE`.
    pub fn sphere() -> Arc<PlanetModel> {
        static M: OnceLock<Arc<PlanetModel>> = OnceLock::new();
        M.get_or_init(|| Arc::new(PlanetModel::new(1.0, 1.0)))
            .clone()
    }

    /// `PlanetModel.WGS84`.
    pub fn wgs84() -> Arc<PlanetModel> {
        static M: OnceLock<Arc<PlanetModel>> = OnceLock::new();
        M.get_or_init(|| Arc::new(PlanetModel::new(6378137.0, 6356752.314245)))
            .clone()
    }

    /// `PlanetModel.CLARKE_1866`.
    pub fn clarke_1866() -> Arc<PlanetModel> {
        static M: OnceLock<Arc<PlanetModel>> = OnceLock::new();
        M.get_or_init(|| Arc::new(PlanetModel::new(6378206.4, 6356583.8)))
            .clone()
    }

    /// `PlanetModel(InputStream)`.
    pub fn read(input: &mut Input<'_>) -> Result<PlanetModel> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        Ok(PlanetModel::new(a, b))
    }

    /// `write(OutputStream)`.
    pub fn write(&self, out: &mut Vec<u8>) {
        write_double(out, self.a);
        write_double(out, self.b);
    }

    /// `isSphere()`.
    pub fn is_sphere(&self) -> bool {
        self.xy_scaling == self.z_scaling
    }

    /// `getMinimumMagnitude()`.
    pub fn minimum_magnitude(&self) -> f64 {
        min(self.xy_scaling, self.z_scaling)
    }

    /// `getMaximumMagnitude()`.
    pub fn maximum_magnitude(&self) -> f64 {
        max(self.xy_scaling, self.z_scaling)
    }

    /// `getMinimumXValue()`.
    pub fn minimum_x_value(&self) -> f64 {
        -self.xy_scaling
    }

    /// `getMaximumXValue()`.
    pub fn maximum_x_value(&self) -> f64 {
        self.xy_scaling
    }

    /// `getMinimumYValue()`.
    pub fn minimum_y_value(&self) -> f64 {
        -self.xy_scaling
    }

    /// `getMaximumYValue()`.
    pub fn maximum_y_value(&self) -> f64 {
        self.xy_scaling
    }

    /// `getMinimumZValue()`.
    pub fn minimum_z_value(&self) -> f64 {
        -self.z_scaling
    }

    /// `getMaximumZValue()`.
    pub fn maximum_z_value(&self) -> f64 {
        self.z_scaling
    }

    /// `getMeanRadius()`.
    pub fn mean_radius(&self) -> f64 {
        self.mean_radius
    }

    /// `encodeValue(x)`: the 32-bit encoding `Geo3DPoint` indexes.
    pub fn encode_value(&self, x: f64) -> Result<i32> {
        use crate::geo::java_double_string as d;
        let mut x = x;
        if x > self.maximum_magnitude() {
            return Err(Error::IllegalArgument(format!(
                "value={} is out-of-bounds (greater than planetMax={})",
                d(x),
                d(self.maximum_magnitude())
            )));
        }
        if x == self.maximum_magnitude() {
            x = next_down(x);
        }
        if x < -self.maximum_magnitude() {
            return Err(Error::IllegalArgument(format!(
                "value={} is out-of-bounds (less than than -planetMax={})",
                d(x),
                d(-self.maximum_magnitude())
            )));
        }
        // `(long) Math.floor(x / DECODE)`, then `(int)`: in range by the checks
        // above (a NaN passes them, and Java's casts make it 0).
        let result = floor(x / self.decode);
        Ok(java_long_to_int(java_double_to_long(result)))
    }

    /// `decodeValue(x)`: the center of the encoded cell (the extremes decode
    /// to `-MAX_VALUE`/`MAX_VALUE` exactly).
    pub fn decode_value(&self, x: i32) -> f64 {
        if x == self.min_encoded_value {
            -self.max_value
        } else if x == self.max_encoded_value {
            self.max_value
        } else {
            (f64::from(x) + 0.5) * self.decode
        }
    }

    /// `getDocValueEncoder()`.
    pub fn doc_value_encoder(&self) -> &DocValueEncoder {
        &self.doc_value_encoder
    }

    /// `pointOnSurface(v)`.
    pub fn point_on_surface(&self, v: &Vector) -> bool {
        self.point_on_surface_xyz(v.x, v.y, v.z)
    }

    /// `pointOnSurface(x, y, z)`.
    pub fn point_on_surface_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        abs(x * x * self.inverse_xy_scaling * self.inverse_xy_scaling
            + y * y * self.inverse_xy_scaling * self.inverse_xy_scaling
            + z * z * self.inverse_z_scaling * self.inverse_z_scaling
            - 1.0)
            < MINIMUM_RESOLUTION
    }

    /// `pointOutside(v)`.
    pub fn point_outside(&self, v: &Vector) -> bool {
        self.point_outside_xyz(v.x, v.y, v.z)
    }

    /// `pointOutside(x, y, z)`.
    pub fn point_outside_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        (x * x + y * y) * self.inverse_xy_scaling * self.inverse_xy_scaling
            + z * z * self.inverse_z_scaling * self.inverse_z_scaling
            - 1.0
            > MINIMUM_RESOLUTION
    }

    /// `createSurfacePoint(vector)`.
    pub fn create_surface_point(&self, v: &Vector) -> GeoPoint {
        self.create_surface_point_xyz(v.x, v.y, v.z)
    }

    /// `createSurfacePoint(x, y, z)`: the surface point along `(x, y, z)`.
    pub fn create_surface_point_xyz(&self, x: f64, y: f64, z: f64) -> GeoPoint {
        let t = sqrt(
            1.0 / (x * x * self.inverse_xy_scaling_squared
                + y * y * self.inverse_xy_scaling_squared
                + z * z * self.inverse_z_scaling_squared),
        );
        GeoPoint::new(t * x, t * y, t * z)
    }

    /// `bisection(pt1, pt2)`: the surface point halfway between, or `None`
    /// when it is undefined.
    pub fn bisection(&self, pt1: &GeoPoint, pt2: &GeoPoint) -> Option<GeoPoint> {
        let a0 = (pt1.x + pt2.x) * 0.5;
        let b0 = (pt1.y + pt2.y) * 0.5;
        let c0 = (pt1.z + pt2.z) * 0.5;
        let denom = self.inverse_xy_scaling_squared * a0 * a0
            + self.inverse_xy_scaling_squared * b0 * b0
            + self.inverse_z_scaling_squared * c0 * c0;
        if denom < MINIMUM_RESOLUTION {
            return None;
        }
        let t = sqrt(1.0 / denom);
        Some(GeoPoint::new(t * a0, t * b0, t * c0))
    }

    /// `surfaceDistance(pt1, pt2)`: Vincenty's inverse formula, in planet
    /// units.
    #[allow(non_snake_case)]
    pub fn surface_distance(&self, pt1: &GeoPoint, pt2: &GeoPoint) -> f64 {
        let L = pt2.longitude() - pt1.longitude();
        let U1 = atan((1.0 - self.scaled_flattening) * tan(pt1.latitude()));
        let U2 = atan((1.0 - self.scaled_flattening) * tan(pt2.latitude()));
        let sinU1 = sin(U1);
        let cosU1 = cos(U1);
        let sinU2 = sin(U2);
        let cosU2 = cos(U2);
        let dCosU1CosU2 = cosU1 * cosU2;
        let dCosU1SinU2 = cosU1 * sinU2;
        let dSinU1SinU2 = sinU1 * sinU2;
        let dSinU1CosU2 = sinU1 * cosU2;
        let mut lambda = L;
        let mut iter_limit = 0;
        let mut cos_sq_alpha;
        let mut sin_sigma;
        let mut cos2_sigma_m;
        let mut cos_sigma;
        let mut sigma;
        loop {
            let sin_lambda = sin(lambda);
            let cos_lambda = cos(lambda);
            sin_sigma = sqrt(
                (cosU2 * sin_lambda) * (cosU2 * sin_lambda)
                    + (dCosU1SinU2 - dSinU1CosU2 * cos_lambda)
                        * (dCosU1SinU2 - dSinU1CosU2 * cos_lambda),
            );
            if sin_sigma == 0.0 {
                return 0.0;
            }
            cos_sigma = dSinU1SinU2 + dCosU1CosU2 * cos_lambda;
            sigma = atan2(sin_sigma, cos_sigma);
            let sin_alpha = dCosU1CosU2 * sin_lambda / sin_sigma;
            cos_sq_alpha = 1.0 - sin_alpha * sin_alpha;
            cos2_sigma_m = cos_sigma - 2.0 * dSinU1SinU2 / cos_sq_alpha;
            if cos2_sigma_m.is_nan() {
                cos2_sigma_m = 0.0; // equatorial line: cosSqAlpha=0
            }
            let c = self.scaled_flattening / 16.0
                * cos_sq_alpha
                * (4.0 + self.scaled_flattening * (4.0 - 3.0 * cos_sq_alpha));
            let lambda_p = lambda;
            lambda = L
                + (1.0 - c)
                    * self.scaled_flattening
                    * sin_alpha
                    * (sigma
                        + c * sin_sigma
                            * (cos2_sigma_m
                                + c * cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)));
            iter_limit += 1;
            if !(abs(lambda - lambda_p) >= MINIMUM_RESOLUTION && iter_limit < 100) {
                break;
            }
        }
        let u_sq = cos_sq_alpha * self.square_ratio;
        let A = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
        let B = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
        let delta_sigma = B
            * sin_sigma
            * (cos2_sigma_m
                + B / 4.0
                    * (cos_sigma * (-1.0 + 2.0 * cos2_sigma_m * cos2_sigma_m)
                        - B / 6.0
                            * cos2_sigma_m
                            * (-3.0 + 4.0 * sin_sigma * sin_sigma)
                            * (-3.0 + 4.0 * cos2_sigma_m * cos2_sigma_m)));
        self.z_scaling * self.inverse_scale * A * (sigma - delta_sigma)
    }

    /// `surfacePointOnBearing(from, dist, bearing)`: Vincenty's direct
    /// formula.
    #[allow(non_snake_case)]
    pub fn surface_point_on_bearing(
        &self,
        from: &GeoPoint,
        dist: f64,
        bearing: f64,
    ) -> Result<GeoPoint> {
        use std::f64::consts::PI;
        let lat = from.latitude();
        let lon = from.longitude();
        let sinalpha1 = sin(bearing);
        let cosalpha1 = cos(bearing);
        let tanU1 = (1.0 - self.scaled_flattening) * tan(lat);
        let cosU1 = 1.0 / sqrt(1.0 + tanU1 * tanU1);
        let sinU1 = tanU1 * cosU1;
        let sigma1 = atan2(tanU1, cosalpha1);
        let sinalpha = cosU1 * sinalpha1;
        let cos_sqalpha = 1.0 - sinalpha * sinalpha;
        let u_sq = cos_sqalpha * self.square_ratio;
        let A = 1.0 + u_sq / 16384.0 * (4096.0 + u_sq * (-768.0 + u_sq * (320.0 - 175.0 * u_sq)));
        let B = u_sq / 1024.0 * (256.0 + u_sq * (-128.0 + u_sq * (74.0 - 47.0 * u_sq)));
        let mut cos2sigma_m;
        let mut sinsigma;
        let mut cossigma;
        let mut sigma = dist / (self.z_scaling * self.inverse_scale * A);
        let mut iterations = 0.0;
        loop {
            cos2sigma_m = cos(2.0 * sigma1 + sigma);
            sinsigma = sin(sigma);
            cossigma = cos(sigma);
            let deltasigma = B
                * sinsigma
                * (cos2sigma_m
                    + B / 4.0
                        * (cossigma * (-1.0 + 2.0 * cos2sigma_m * cos2sigma_m)
                            - B / 6.0
                                * cos2sigma_m
                                * (-3.0 + 4.0 * sinsigma * sinsigma)
                                * (-3.0 + 4.0 * cos2sigma_m * cos2sigma_m)));
            let sigmaprime = sigma;
            sigma = dist / (self.z_scaling * self.inverse_scale * A) + deltasigma;
            iterations += 1.0;
            if !(abs(sigma - sigmaprime) >= MINIMUM_RESOLUTION && iterations < 100.0) {
                break;
            }
        }
        let x = sinU1 * sinsigma - cosU1 * cossigma * cosalpha1;
        let phi2 = atan2(
            sinU1 * cossigma + cosU1 * sinsigma * cosalpha1,
            (1.0 - self.scaled_flattening) * sqrt(sinalpha * sinalpha + x * x),
        );
        let lambda = atan2(
            sinsigma * sinalpha1,
            cosU1 * cossigma - sinU1 * sinsigma * cosalpha1,
        );
        let C = self.scaled_flattening / 16.0
            * cos_sqalpha
            * (4.0 + self.scaled_flattening * (4.0 - 3.0 * cos_sqalpha));
        let L = lambda
            - (1.0 - C)
                * self.scaled_flattening
                * sinalpha
                * (sigma
                    + C * sinsigma
                        * (cos2sigma_m + C * cossigma * (-1.0 + 2.0 * cos2sigma_m * cos2sigma_m)));
        let lambda2 = rem(lon + L + 3.0 * PI, 2.0 * PI) - PI; // normalise to -180..+180
        GeoPoint::from_lat_lon(self, phi2, lambda2)
    }

    /// `hashCode()`.
    pub fn java_hash_code(&self) -> i32 {
        super::jmath::double_hash(self.a).wrapping_add(super::jmath::double_hash(self.b))
    }

    /// `PlanetModel.SPHERE`, `.WGS84`, `.CLARKE_1866` or the axes, as Java's
    /// `toString`.
    pub fn java_to_string(&self) -> String {
        if *self == *PlanetModel::sphere() {
            "PlanetModel.SPHERE".into()
        } else if *self == *PlanetModel::wgs84() {
            "PlanetModel.WGS84".into()
        } else if *self == *PlanetModel::clarke_1866() {
            "PlanetModel.CLARKE_1866".into()
        } else {
            use crate::geo::java_double_string as d;
            format!(
                "PlanetModel(xyScaling={} zScaling={})",
                d(self.a),
                d(self.b)
            )
        }
    }
}

/// `getNextSafeDouble(x)`: the next double up from `x` whose bottom 32 bits
/// are clear.
fn next_safe_double(x: f64) -> f64 {
    let mut bits = super::jmath::double_to_long_bits(x);
    bits = bits.wrapping_add(i64::from(i32::MAX));
    bits &= !i64::from(i32::MAX);
    f64::from_bits(bits as u64)
}

/// Java's `(long) d`: saturating, NaN to 0.
#[inline]
pub(crate) fn java_double_to_long(d: f64) -> i64 {
    d as i64
}

/// Java's `(int) l`: the low 32 bits.
#[inline]
pub(crate) fn java_long_to_int(l: i64) -> i32 {
    l as i32
}

/// `PlanetModel.DocValueEncoder`: a point packed into one `long`, 21 bits
/// per dimension, as `Geo3DDocValuesField` stores it.
#[derive(Debug, Clone, Default)]
pub struct DocValueEncoder {
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
    inverse_x_factor: f64,
    inverse_y_factor: f64,
    inverse_z_factor: f64,
    x_factor: f64,
    y_factor: f64,
    z_factor: f64,
    x_step: f64,
    y_step: f64,
    z_step: f64,
}

const INVERSE_MAXIMUM_VALUE: f64 = 1.0 / (0x1F_FFFF as f64);
/// Fudge factor for step adjustments (LUCENE-7430).
const STEP_FUDGE: f64 = 10.0;

impl DocValueEncoder {
    fn new(pm: &PlanetModel) -> DocValueEncoder {
        let inverse_x_factor =
            (pm.maximum_x_value() - pm.minimum_x_value()) * INVERSE_MAXIMUM_VALUE;
        let inverse_y_factor =
            (pm.maximum_y_value() - pm.minimum_y_value()) * INVERSE_MAXIMUM_VALUE;
        let inverse_z_factor =
            (pm.maximum_z_value() - pm.minimum_z_value()) * INVERSE_MAXIMUM_VALUE;
        DocValueEncoder {
            min_x: pm.minimum_x_value(),
            max_x: pm.maximum_x_value(),
            min_y: pm.minimum_y_value(),
            max_y: pm.maximum_y_value(),
            min_z: pm.minimum_z_value(),
            max_z: pm.maximum_z_value(),
            inverse_x_factor,
            inverse_y_factor,
            inverse_z_factor,
            x_factor: 1.0 / inverse_x_factor,
            y_factor: 1.0 / inverse_y_factor,
            z_factor: 1.0 / inverse_z_factor,
            x_step: inverse_x_factor * STEP_FUDGE,
            y_step: inverse_y_factor * STEP_FUDGE,
            z_step: inverse_z_factor * STEP_FUDGE,
        }
    }

    /// `encodePoint(point)`.
    pub fn encode_point(&self, point: &Vector) -> Result<i64> {
        self.encode_point_xyz(point.x, point.y, point.z)
    }

    /// `encodePoint(x, y, z)`.
    pub fn encode_point_xyz(&self, x: f64, y: f64, z: f64) -> Result<i64> {
        let x_encoded = self.encode_x(x)?;
        let y_encoded = self.encode_y(y)?;
        let z_encoded = self.encode_z(z)?;
        Ok((i64::from(x_encoded & 0x1F_FFFF) << 42)
            | (i64::from(y_encoded & 0x1F_FFFF) << 21)
            | i64::from(z_encoded & 0x1F_FFFF))
    }

    /// `decodePoint(docValue)`.
    pub fn decode_point(&self, doc_value: i64) -> GeoPoint {
        GeoPoint::new(
            self.decode_x_value(doc_value),
            self.decode_y_value(doc_value),
            self.decode_z_value(doc_value),
        )
    }

    /// `decodeXValue(docValue)`.
    pub fn decode_x_value(&self, doc_value: i64) -> f64 {
        self.decode_x(((doc_value >> 42) as i32) & 0x1F_FFFF)
    }

    /// `decodeYValue(docValue)`.
    pub fn decode_y_value(&self, doc_value: i64) -> f64 {
        self.decode_y(((doc_value >> 21) as i32) & 0x1F_FFFF)
    }

    /// `decodeZValue(docValue)`.
    pub fn decode_z_value(&self, doc_value: i64) -> f64 {
        self.decode_z((doc_value as i32) & 0x1F_FFFF)
    }

    /// `roundDownX`.
    pub fn round_down_x(&self, v: f64) -> f64 {
        v - self.x_step
    }

    /// `roundUpX`.
    pub fn round_up_x(&self, v: f64) -> f64 {
        v + self.x_step
    }

    /// `roundDownY`.
    pub fn round_down_y(&self, v: f64) -> f64 {
        v - self.y_step
    }

    /// `roundUpY`.
    pub fn round_up_y(&self, v: f64) -> f64 {
        v + self.y_step
    }

    /// `roundDownZ`.
    pub fn round_down_z(&self, v: f64) -> f64 {
        v - self.z_step
    }

    /// `roundUpZ`.
    pub fn round_up_z(&self, v: f64) -> f64 {
        v + self.z_step
    }

    fn encode_x(&self, x: f64) -> Result<i32> {
        if x > self.max_x {
            return Err(Error::IllegalArgument(
                "x value exceeds planet model maximum".into(),
            ));
        } else if x < self.min_x {
            return Err(Error::IllegalArgument(
                "x value less than planet model minimum".into(),
            ));
        }
        Ok(floor((x - self.min_x) * self.x_factor + 0.5) as i32)
    }

    fn decode_x(&self, x: i32) -> f64 {
        f64::from(x) * self.inverse_x_factor + self.min_x
    }

    fn encode_y(&self, y: f64) -> Result<i32> {
        if y > self.max_y {
            return Err(Error::IllegalArgument(
                "y value exceeds planet model maximum".into(),
            ));
        } else if y < self.min_y {
            return Err(Error::IllegalArgument(
                "y value less than planet model minimum".into(),
            ));
        }
        Ok(floor((y - self.min_y) * self.y_factor + 0.5) as i32)
    }

    fn decode_y(&self, y: i32) -> f64 {
        f64::from(y) * self.inverse_y_factor + self.min_y
    }

    fn encode_z(&self, z: f64) -> Result<i32> {
        if z > self.max_z {
            return Err(Error::IllegalArgument(
                "z value exceeds planet model maximum".into(),
            ));
        } else if z < self.min_z {
            return Err(Error::IllegalArgument(
                "z value less than planet model minimum".into(),
            ));
        }
        Ok(floor((z - self.min_z) * self.z_factor + 0.5) as i32)
    }

    fn decode_z(&self, z: i32) -> f64 {
        f64::from(z) * self.inverse_z_factor + self.min_z
    }
}
