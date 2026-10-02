//! `GeoLatitudeZone` (`org.apache.lucene.spatial3d.geom.GeoLatitudeZone`),
//! `GeoNorthLatitudeZone` and `GeoSouthLatitudeZone`: every longitude
//! between two latitudes, above one, or below one.

use super::geo_bbox_factory::make_geo_bbox;
use super::prelude::*;

/// `planePoints`: the zones have no notable points.
const PLANE_POINTS: &[GeoPoint] = &[];

/// The point at `sinLat` on the prime meridian, as the zones build it:
/// `GeoPoint(planetModel, sinLat, 0.0, sqrt(1 - sinLat^2), 1.0)`.
fn meridian_point(pm: &PlanetModel, sin_lat: f64) -> GeoPoint {
    GeoPoint::from_trig(pm, sin_lat, 0.0, sqrt(1.0 - sin_lat * sin_lat), 1.0)
}

/// Between two latitudes.
#[derive(Debug, Clone)]
pub struct GeoLatitudeZone {
    planet_model: Arc<PlanetModel>,
    top_lat: f64,
    bottom_lat: f64,
    cos_top_lat: f64,
    cos_bottom_lat: f64,
    top_plane: SidedPlane,
    bottom_plane: SidedPlane,
    interior_point: GeoPoint,
    edge_points: [GeoPoint; 2],
}

impl GeoLatitudeZone {
    /// `GeoLatitudeZone(planetModel, topLat, bottomLat)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        top_lat: f64,
        bottom_lat: f64,
    ) -> Result<GeoLatitudeZone> {
        let pm = &**planet_model;
        let sin_top_lat = sin(top_lat);
        let sin_bottom_lat = sin(bottom_lat);
        let cos_top_lat = cos(top_lat);
        let cos_bottom_lat = cos(bottom_lat);
        let middle_lat = (top_lat + bottom_lat) * 0.5;
        let sin_middle_lat = sin(middle_lat);
        let interior_point = meridian_point(pm, sin_middle_lat);
        let top_boundary_point = meridian_point(pm, sin_top_lat);
        let bottom_boundary_point = meridian_point(pm, sin_bottom_lat);
        let top_plane = SidedPlane::horizontal(&interior_point, pm, sin_top_lat)?;
        let bottom_plane = SidedPlane::horizontal(&interior_point, pm, sin_bottom_lat)?;
        Ok(GeoLatitudeZone {
            planet_model: planet_model.clone(),
            top_lat,
            bottom_lat,
            cos_top_lat,
            cos_bottom_lat,
            top_plane,
            bottom_plane,
            interior_point,
            edge_points: [top_boundary_point, bottom_boundary_point],
        })
    }

    /// `GeoLatitudeZone(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<GeoLatitudeZone> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        GeoLatitudeZone::new(planet_model, a, b)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let top_distance =
            style.compute_distance_to_plane(pm, &self.top_plane, x, y, z, &[&self.bottom_plane]);
        let bottom_distance =
            style.compute_distance_to_plane(pm, &self.bottom_plane, x, y, z, &[&self.top_plane]);
        min(top_distance, bottom_distance)
    }
}

impl SerializableObject for GeoLatitudeZone {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.top_lat);
        write_double(out, self.bottom_lat);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(15)
    }
}

impl_planet_object!(GeoLatitudeZone);
impl_membership_shape!(GeoLatitudeZone);
impl_base_area!(GeoLatitudeZone);

impl Membership for GeoLatitudeZone {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.top_plane.is_within_xyz(x, y, z) && self.bottom_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoLatitudeZone {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .no_longitude_bound()
            .add_horizontal_plane(pm, self.top_lat, &self.top_plane, &[])
            .add_horizontal_plane(pm, self.bottom_lat, &self.bottom_plane, &[]);
    }
}

impl GeoShape for GeoLatitudeZone {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        let pm = &*self.planet_model;
        p.intersects(
            pm,
            &self.top_plane,
            notable_points,
            PLANE_POINTS,
            bounds,
            &[&self.bottom_plane],
        ) || p.intersects(
            pm,
            &self.bottom_plane,
            notable_points,
            PLANE_POINTS,
            bounds,
            &[&self.top_plane],
        )
    }
}

impl GeoAreaShape for GeoLatitudeZone {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.top_plane, PLANE_POINTS, &[&self.bottom_plane])
            || geo_shape.intersects(&self.bottom_plane, PLANE_POINTS, &[&self.top_plane])
    }
}

impl GeoSizeable for GeoLatitudeZone {
    fn radius(&self) -> f64 {
        // If the zone straddles the equator, the max distance is the full
        // width of the planet.
        if self.top_lat > 0.0 && self.bottom_lat < 0.0 {
            return PI;
        }
        let mut max_cos_lat = self.cos_top_lat;
        if max_cos_lat < self.cos_bottom_lat {
            max_cos_lat = self.cos_bottom_lat;
        }
        max_cos_lat * PI
    }

    fn center(&self) -> GeoPoint {
        self.interior_point.clone()
    }
}

impl GeoBBox for GeoLatitudeZone {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        make_geo_bbox(
            &self.planet_model,
            self.top_lat + angle,
            self.bottom_lat - angle,
            -PI,
            PI,
        )
    }
}

/// Above a latitude, to the north pole.
#[derive(Debug, Clone)]
pub struct GeoNorthLatitudeZone {
    planet_model: Arc<PlanetModel>,
    bottom_lat: f64,
    cos_bottom_lat: f64,
    bottom_plane: SidedPlane,
    interior_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoNorthLatitudeZone {
    /// `GeoNorthLatitudeZone(planetModel, bottomLat)`.
    pub fn new(planet_model: &Arc<PlanetModel>, bottom_lat: f64) -> Result<GeoNorthLatitudeZone> {
        let pm = &**planet_model;
        let sin_bottom_lat = sin(bottom_lat);
        let cos_bottom_lat = cos(bottom_lat);
        let middle_lat = (PI * 0.5 + bottom_lat) * 0.5;
        let sin_middle_lat = sin(middle_lat);
        let interior_point = meridian_point(pm, sin_middle_lat);
        let bottom_boundary_point = meridian_point(pm, sin_bottom_lat);
        let bottom_plane = SidedPlane::horizontal(&interior_point, pm, sin_bottom_lat)?;
        Ok(GeoNorthLatitudeZone {
            planet_model: planet_model.clone(),
            bottom_lat,
            cos_bottom_lat,
            bottom_plane,
            interior_point,
            edge_points: [bottom_boundary_point],
        })
    }

    /// `GeoNorthLatitudeZone(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoNorthLatitudeZone> {
        let a = read_double(input)?;
        GeoNorthLatitudeZone::new(planet_model, a)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance_to_plane(&self.planet_model, &self.bottom_plane, x, y, z, &[])
    }
}

impl SerializableObject for GeoNorthLatitudeZone {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.bottom_lat);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(17)
    }
}

impl_planet_object!(GeoNorthLatitudeZone);
impl_membership_shape!(GeoNorthLatitudeZone);
impl_base_area!(GeoNorthLatitudeZone);

impl Membership for GeoNorthLatitudeZone {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.bottom_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoNorthLatitudeZone {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds.add_horizontal_plane(pm, self.bottom_lat, &self.bottom_plane, &[]);
    }
}

impl GeoShape for GeoNorthLatitudeZone {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        p.intersects(
            &self.planet_model,
            &self.bottom_plane,
            notable_points,
            PLANE_POINTS,
            bounds,
            &[],
        )
    }
}

impl GeoAreaShape for GeoNorthLatitudeZone {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.bottom_plane, PLANE_POINTS, &[])
    }
}

impl GeoSizeable for GeoNorthLatitudeZone {
    fn radius(&self) -> f64 {
        if self.bottom_lat < 0.0 {
            return PI;
        }
        self.cos_bottom_lat * PI
    }

    fn center(&self) -> GeoPoint {
        self.interior_point.clone()
    }
}

impl GeoBBox for GeoNorthLatitudeZone {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        make_geo_bbox(
            &self.planet_model,
            PI * 0.5,
            self.bottom_lat - angle,
            -PI,
            PI,
        )
    }
}

/// Below a latitude, to the south pole.
#[derive(Debug, Clone)]
pub struct GeoSouthLatitudeZone {
    planet_model: Arc<PlanetModel>,
    top_lat: f64,
    cos_top_lat: f64,
    top_plane: SidedPlane,
    interior_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoSouthLatitudeZone {
    /// `GeoSouthLatitudeZone(planetModel, topLat)`.
    pub fn new(planet_model: &Arc<PlanetModel>, top_lat: f64) -> Result<GeoSouthLatitudeZone> {
        let pm = &**planet_model;
        let sin_top_lat = sin(top_lat);
        let cos_top_lat = cos(top_lat);
        let middle_lat = (top_lat - PI * 0.5) * 0.5;
        let sin_middle_lat = sin(middle_lat);
        let interior_point = meridian_point(pm, sin_middle_lat);
        let top_boundary_point = meridian_point(pm, sin_top_lat);
        let top_plane = SidedPlane::horizontal(&interior_point, pm, sin_top_lat)?;
        Ok(GeoSouthLatitudeZone {
            planet_model: planet_model.clone(),
            top_lat,
            cos_top_lat,
            top_plane,
            interior_point,
            edge_points: [top_boundary_point],
        })
    }

    /// `GeoSouthLatitudeZone(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoSouthLatitudeZone> {
        let a = read_double(input)?;
        GeoSouthLatitudeZone::new(planet_model, a)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance_to_plane(&self.planet_model, &self.top_plane, x, y, z, &[])
    }
}

impl SerializableObject for GeoSouthLatitudeZone {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.top_lat);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(19)
    }
}

impl_planet_object!(GeoSouthLatitudeZone);
impl_membership_shape!(GeoSouthLatitudeZone);
impl_base_area!(GeoSouthLatitudeZone);

impl Membership for GeoSouthLatitudeZone {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.top_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoSouthLatitudeZone {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds.add_horizontal_plane(pm, self.top_lat, &self.top_plane, &[]);
    }
}

impl GeoShape for GeoSouthLatitudeZone {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        p.intersects(
            &self.planet_model,
            &self.top_plane,
            notable_points,
            PLANE_POINTS,
            bounds,
            &[],
        )
    }
}

impl GeoAreaShape for GeoSouthLatitudeZone {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.top_plane, PLANE_POINTS, &[])
    }
}

impl GeoSizeable for GeoSouthLatitudeZone {
    fn radius(&self) -> f64 {
        if self.top_lat > 0.0 {
            return PI;
        }
        self.cos_top_lat * PI
    }

    fn center(&self) -> GeoPoint {
        self.interior_point.clone()
    }
}

impl GeoBBox for GeoSouthLatitudeZone {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        make_geo_bbox(&self.planet_model, self.top_lat + angle, -PI * 0.5, -PI, PI)
    }
}
