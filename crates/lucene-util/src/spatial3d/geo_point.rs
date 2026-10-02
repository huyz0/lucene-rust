//! `GeoPoint` (`org.apache.lucene.spatial3d.geom.GeoPoint`): a point on (or
//! near) the planet's surface, with its magnitude, latitude and longitude
//! computed lazily and cached as Java caches them in `volatile` fields.

use std::sync::atomic::{AtomicU64, Ordering};

use super::jmath::{abs, asin, atan2, sin_cos};
use super::planet_model::PlanetModel;
use super::serializable::{read_double, write_double, Input};
use super::tools::safe_acos;
use super::vector::{Vector, MINIMUM_RESOLUTION};
use super::{Error, Result};

/// A lazily computed `double`, `NEGATIVE_INFINITY` until first read. Relaxed
/// atomics: the value is a pure function of the point, so a race only
/// computes it twice, as Java's unsynchronized `volatile` caches do.
pub(crate) struct LazyDouble(AtomicU64);

impl LazyDouble {
    const UNSET: u64 = 0xfff0_0000_0000_0000; // f64::NEG_INFINITY

    fn unset() -> LazyDouble {
        LazyDouble(AtomicU64::new(Self::UNSET))
    }

    fn of(v: f64) -> LazyDouble {
        LazyDouble(AtomicU64::new(v.to_bits()))
    }

    #[inline]
    fn get(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    #[inline]
    fn set(&self, v: f64) {
        self.0.store(v.to_bits(), Ordering::Relaxed)
    }
}

impl Clone for LazyDouble {
    fn clone(&self) -> Self {
        LazyDouble(AtomicU64::new(self.0.load(Ordering::Relaxed)))
    }
}

/// A point, as a [`Vector`] from the planet's center.
#[derive(Clone)]
pub struct GeoPoint {
    v: Vector,
    magnitude: LazyDouble,
    latitude: LazyDouble,
    longitude: LazyDouble,
}

impl std::ops::Deref for GeoPoint {
    type Target = Vector;

    #[inline]
    fn deref(&self) -> &Vector {
        &self.v
    }
}

impl std::fmt::Debug for GeoPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl std::fmt::Display for GeoPoint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.longitude.get() == f64::NEG_INFINITY {
            return write!(f, "{}", self.v);
        }
        use crate::geo::java_double_string as d;
        write!(
            f,
            "[lat={}, lon={}({})]",
            d(self.latitude()),
            d(self.longitude()),
            self.v
        )
    }
}

/// `Vector.equals`: exact coordinates (`==`, so `0.0 == -0.0`).
impl super::shape::SerializableObject for GeoPoint {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        GeoPoint::write(self, out);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(0)
    }
}

impl PartialEq for GeoPoint {
    fn eq(&self, other: &GeoPoint) -> bool {
        self.v == other.v
    }
}

impl GeoPoint {
    /// `GeoPoint(x, y, z)`.
    #[inline]
    pub fn new(x: f64, y: f64, z: f64) -> GeoPoint {
        GeoPoint {
            v: Vector::new(x, y, z),
            magnitude: LazyDouble::unset(),
            latitude: LazyDouble::unset(),
            longitude: LazyDouble::unset(),
        }
    }

    /// `GeoPoint(Vector)`-equivalent: a point at `v`'s coordinates.
    #[inline]
    pub fn from_vector(v: &Vector) -> GeoPoint {
        GeoPoint::new(v.x, v.y, v.z)
    }

    /// `GeoPoint(magnitude, x, y, z)`: `(x, y, z)` scaled by `magnitude`.
    pub fn with_magnitude(magnitude: f64, x: f64, y: f64, z: f64) -> GeoPoint {
        GeoPoint {
            v: Vector::new(x * magnitude, y * magnitude, z * magnitude),
            magnitude: LazyDouble::of(magnitude),
            latitude: LazyDouble::unset(),
            longitude: LazyDouble::unset(),
        }
    }

    /// `GeoPoint(magnitude, x, y, z, lat, lon)`.
    pub fn with_magnitude_lat_lon(
        magnitude: f64,
        x: f64,
        y: f64,
        z: f64,
        lat: f64,
        lon: f64,
    ) -> Result<GeoPoint> {
        use crate::geo::java_double_string as d;
        use std::f64::consts::PI;
        if lat > PI * 0.5 || lat < -PI * 0.5 {
            return Err(Error::IllegalArgument(format!(
                "Latitude {} is out of range: must range from -Math.PI/2 to Math.PI/2",
                d(lat)
            )));
        }
        if lon < -PI || lon > PI {
            return Err(Error::IllegalArgument(format!(
                "Longitude {} is out of range: must range from -Math.PI to Math.PI",
                d(lon)
            )));
        }
        Ok(GeoPoint {
            v: Vector::new(x * magnitude, y * magnitude, z * magnitude),
            magnitude: LazyDouble::of(magnitude),
            latitude: LazyDouble::of(lat),
            longitude: LazyDouble::of(lon),
        })
    }

    /// `GeoPoint(lat, lon, x, y, z)`: coordinates with known lat/lon.
    pub fn with_lat_lon_xyz(lat: f64, lon: f64, x: f64, y: f64, z: f64) -> GeoPoint {
        GeoPoint {
            v: Vector::new(x, y, z),
            magnitude: LazyDouble::unset(),
            latitude: LazyDouble::of(lat),
            longitude: LazyDouble::of(lon),
        }
    }

    /// `GeoPoint(planetModel, sinLat, sinLon, cosLat, cosLon, lat, lon)`.
    pub fn from_trig_lat_lon(
        pm: &PlanetModel,
        sin_lat: f64,
        sin_lon: f64,
        cos_lat: f64,
        cos_lon: f64,
        lat: f64,
        lon: f64,
    ) -> Result<GeoPoint> {
        GeoPoint::with_magnitude_lat_lon(
            Vector::compute_desired_ellipsoid_magnitude(
                pm,
                cos_lat * cos_lon,
                cos_lat * sin_lon,
                sin_lat,
            ),
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            sin_lat,
            lat,
            lon,
        )
    }

    /// `GeoPoint(planetModel, sinLat, sinLon, cosLat, cosLon)`.
    pub fn from_trig(
        pm: &PlanetModel,
        sin_lat: f64,
        sin_lon: f64,
        cos_lat: f64,
        cos_lon: f64,
    ) -> GeoPoint {
        GeoPoint::with_magnitude(
            Vector::compute_desired_ellipsoid_magnitude(
                pm,
                cos_lat * cos_lon,
                cos_lat * sin_lon,
                sin_lat,
            ),
            cos_lat * cos_lon,
            cos_lat * sin_lon,
            sin_lat,
        )
    }

    /// `GeoPoint(planetModel, lat, lon)`: the surface point at a latitude
    /// and longitude in radians. Fails as Java does when either is out of
    /// range.
    pub fn from_lat_lon(pm: &PlanetModel, lat: f64, lon: f64) -> Result<GeoPoint> {
        let (sin_lat, cos_lat) = sin_cos(lat);
        let (sin_lon, cos_lon) = sin_cos(lon);
        GeoPoint::from_trig_lat_lon(pm, sin_lat, sin_lon, cos_lat, cos_lon, lat, lon)
    }

    /// `GeoPoint(InputStream)`: latitude, longitude, x, y, z.
    pub fn read(input: &mut Input<'_>) -> Result<GeoPoint> {
        let lat = read_double(input)?;
        let lon = read_double(input)?;
        let x = read_double(input)?;
        let y = read_double(input)?;
        let z = read_double(input)?;
        Ok(GeoPoint::with_lat_lon_xyz(lat, lon, x, y, z))
    }

    /// `write(OutputStream)`.
    pub fn write(&self, out: &mut Vec<u8>) {
        write_double(out, self.latitude());
        write_double(out, self.longitude());
        write_double(out, self.x);
        write_double(out, self.y);
        write_double(out, self.z);
    }

    /// The point as a plain [`Vector`].
    #[inline]
    pub fn vector(&self) -> &Vector {
        &self.v
    }

    /// `arcDistance(Vector)` for a `GeoPoint` argument: Java's virtual
    /// `v.magnitude()` reads the other point's cached magnitude, which for a
    /// point built from a magnitude is that magnitude, not the coordinates'
    /// recomputed length.
    pub fn arc_distance(&self, v: &GeoPoint) -> f64 {
        safe_acos(self.v.dot_product(v) / (self.magnitude() * v.magnitude()))
    }

    /// `arcDistance(Vector)` for a plain vector.
    pub fn arc_distance_vector(&self, v: &Vector) -> f64 {
        safe_acos(self.v.dot_product(v) / (self.magnitude() * v.magnitude()))
    }

    /// `normalize()`, through this point's (cached) magnitude.
    pub fn normalize(&self) -> Option<Vector> {
        let denom = self.magnitude();
        if denom < MINIMUM_RESOLUTION {
            return None;
        }
        let norm_factor = 1.0 / denom;
        Some(Vector::new(
            self.x * norm_factor,
            self.y * norm_factor,
            self.z * norm_factor,
        ))
    }

    /// `arcDistance(x, y, z)`.
    pub fn arc_distance_xyz(&self, x: f64, y: f64, z: f64) -> f64 {
        safe_acos(
            self.v.dot_product_xyz(x, y, z) / (self.magnitude() * Vector::magnitude_of(x, y, z)),
        )
    }

    /// `getLatitude()`.
    pub fn latitude(&self) -> f64 {
        let lat = self.latitude.get();
        if lat == f64::NEG_INFINITY {
            let lat = asin(self.v.z / self.magnitude());
            self.latitude.set(lat);
            return lat;
        }
        lat
    }

    /// `getLongitude()`.
    pub fn longitude(&self) -> f64 {
        let lon = self.longitude.get();
        if lon == f64::NEG_INFINITY {
            let lon = if abs(self.v.x) < MINIMUM_RESOLUTION && abs(self.v.y) < MINIMUM_RESOLUTION {
                0.0
            } else {
                atan2(self.v.y, self.v.x)
            };
            self.longitude.set(lon);
            return lon;
        }
        lon
    }

    /// `magnitude()` (overridden: cached).
    pub fn magnitude(&self) -> f64 {
        let mag = self.magnitude.get();
        if mag == f64::NEG_INFINITY {
            let mag = self.v.magnitude();
            self.magnitude.set(mag);
            return mag;
        }
        mag
    }

    /// `isIdentical(GeoPoint)`.
    pub fn is_identical(&self, p: &Vector) -> bool {
        self.is_identical_xyz(p.x, p.y, p.z)
    }

    /// `isIdentical(x, y, z)`.
    pub fn is_identical_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        abs(self.v.x - x) < MINIMUM_RESOLUTION
            && abs(self.v.y - y) < MINIMUM_RESOLUTION
            && abs(self.v.z - z) < MINIMUM_RESOLUTION
    }

    /// `Vector.hashCode()`.
    pub fn java_hash_code(&self) -> i32 {
        self.v.java_hash_code()
    }
}
