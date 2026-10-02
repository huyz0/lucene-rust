//! `GeoDegenerateVerticalLine`
//! (`org.apache.lucene.spatial3d.geom.GeoDegenerateVerticalLine`),
//! `GeoDegenerateLongitudeSlice` and `GeoDegenerateLatitudeZone`: a segment
//! of one meridian, a whole meridian, and a whole latitude.

#![allow(non_snake_case)]

use super::geo_bbox_factory::make_geo_bbox;
use super::geo_degenerate_horizontal_line::line_relationship;
use super::prelude::*;

/// A segment of a meridian.
#[derive(Debug, Clone)]
pub struct GeoDegenerateVerticalLine {
    planet_model: Arc<PlanetModel>,
    top_lat: f64,
    bottom_lat: f64,
    longitude: f64,
    UHC: GeoPoint,
    LHC: GeoPoint,
    top_plane: SidedPlane,
    bottom_plane: SidedPlane,
    bounding_plane: SidedPlane,
    plane: Plane,
    plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoDegenerateVerticalLine {
    /// `GeoDegenerateVerticalLine(planetModel, topLat, bottomLat, longitude)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        top_lat: f64,
        bottom_lat: f64,
        longitude: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if top_lat > PI * 0.5 || top_lat < -PI * 0.5 {
            return Err(illegal("Top latitude out of range"));
        }
        if bottom_lat > PI * 0.5 || bottom_lat < -PI * 0.5 {
            return Err(illegal("Bottom latitude out of range"));
        }
        if top_lat < bottom_lat {
            return Err(illegal("Top latitude less than bottom latitude"));
        }
        if longitude < -PI || longitude > PI {
            return Err(illegal("Longitude out of range"));
        }
        let sin_top_lat = sin(top_lat);
        let cos_top_lat = cos(top_lat);
        let sin_bottom_lat = sin(bottom_lat);
        let cos_bottom_lat = cos(bottom_lat);
        let sin_longitude = sin(longitude);
        let cos_longitude = cos(longitude);
        let UHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_top_lat,
            sin_longitude,
            cos_top_lat,
            cos_longitude,
            top_lat,
            longitude,
        );
        let UHC = UHC?;
        let LHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_bottom_lat,
            sin_longitude,
            cos_bottom_lat,
            cos_longitude,
            bottom_lat,
            longitude,
        );
        let LHC = LHC?;
        let plane = Plane::vertical(cos_longitude, sin_longitude);
        let middle_lat = (top_lat + bottom_lat) * 0.5;
        let sin_middle_lat = sin(middle_lat);
        let cos_middle_lat = cos(middle_lat);
        let center_point = GeoPoint::from_trig(
            pm,
            sin_middle_lat,
            sin_longitude,
            cos_middle_lat,
            cos_longitude,
        );
        let top_plane = SidedPlane::horizontal(&LHC, pm, sin_top_lat)?;
        let bottom_plane = SidedPlane::horizontal(&UHC, pm, sin_bottom_lat)?;
        let bounding_plane = SidedPlane::vertical(&center_point, -sin_longitude, cos_longitude)?;
        Ok(GeoDegenerateVerticalLine {
            planet_model: planet_model.clone(),
            top_lat,
            bottom_lat,
            longitude,
            plane_points: [UHC.clone(), LHC.clone()],
            edge_points: [center_point.clone()],
            UHC,
            LHC,
            top_plane,
            bottom_plane,
            bounding_plane,
            plane,
            center_point,
        })
    }

    /// `GeoDegenerateVerticalLine(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        let c = read_double(input)?;
        Self::new(planet_model, a, b, c)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let distance = style.compute_distance_to_plane(
            pm,
            &self.plane,
            x,
            y,
            z,
            &[&self.top_plane, &self.bottom_plane, &self.bounding_plane],
        );
        let uhc_distance = style.compute_distance(&self.UHC, x, y, z);
        let lhc_distance = style.compute_distance(&self.LHC, x, y, z);
        min(distance, min(uhc_distance, lhc_distance))
    }
}

impl SerializableObject for GeoDegenerateVerticalLine {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.top_lat);
        write_double(out, self.bottom_lat);
        write_double(out, self.longitude);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(14)
    }
}

impl_planet_object!(GeoDegenerateVerticalLine);
impl_membership_shape!(GeoDegenerateVerticalLine);

impl GeoArea for GeoDegenerateVerticalLine {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(line_relationship(self, &self.center_point, path))
    }
}

impl Membership for GeoDegenerateVerticalLine {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane.evaluate_is_zero_xyz(x, y, z)
            && self.bounding_plane.is_within_xyz(x, y, z)
            && self.top_plane.is_within_xyz(x, y, z)
            && self.bottom_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoDegenerateVerticalLine {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_vertical_plane(
                pm,
                self.longitude,
                &self.plane,
                &[&self.bounding_plane, &self.top_plane, &self.bottom_plane],
            )
            .add_point(&self.UHC)
            .add_point(&self.LHC);
    }
}

impl GeoShape for GeoDegenerateVerticalLine {
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
            &self.plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[&self.bounding_plane, &self.top_plane, &self.bottom_plane],
        )
    }
}

impl GeoAreaShape for GeoDegenerateVerticalLine {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.plane,
            &self.plane_points,
            &[&self.bounding_plane, &self.top_plane, &self.bottom_plane],
        )
    }
}

impl GeoSizeable for GeoDegenerateVerticalLine {
    fn radius(&self) -> f64 {
        let top_angle = self.center_point.arc_distance(&self.UHC);
        let bottom_angle = self.center_point.arc_distance(&self.LHC);
        max(top_angle, bottom_angle)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoDegenerateVerticalLine {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        let new_top_lat = self.top_lat + angle;
        let new_bottom_lat = self.bottom_lat - angle;
        let mut new_left_lon = self.longitude - angle;
        let mut new_right_lon = self.longitude + angle;
        let current_lon_span = 2.0 * angle;
        if current_lon_span + 2.0 * angle >= PI * 2.0 {
            new_left_lon = -PI;
            new_right_lon = PI;
        }
        make_geo_bbox(
            &self.planet_model,
            new_top_lat,
            new_bottom_lat,
            new_left_lon,
            new_right_lon,
        )
    }
}

/// A whole meridian, pole to pole.
#[derive(Debug, Clone)]
pub struct GeoDegenerateLongitudeSlice {
    planet_model: Arc<PlanetModel>,
    longitude: f64,
    bounding_plane: SidedPlane,
    plane: Plane,
    interior_point: GeoPoint,
    edge_points: [GeoPoint; 1],
    plane_points: [GeoPoint; 2],
}

impl GeoDegenerateLongitudeSlice {
    /// `GeoDegenerateLongitudeSlice(planetModel, longitude)`.
    pub fn new(planet_model: &Arc<PlanetModel>, longitude: f64) -> Result<Self> {
        let pm = &**planet_model;
        if longitude < -PI || longitude > PI {
            return Err(illegal("Longitude out of range"));
        }
        let sin_longitude = sin(longitude);
        let cos_longitude = cos(longitude);
        let plane = Plane::vertical(cos_longitude, sin_longitude);
        // We need a bounding plane too, which is perpendicular to the
        // longitude plane and sided so that the point (0.0, longitude) is
        // inside.
        let interior_point = GeoPoint::from_trig(pm, 0.0, sin_longitude, 1.0, cos_longitude);
        let bounding_plane = SidedPlane::vertical(&interior_point, -sin_longitude, cos_longitude)?;
        Ok(GeoDegenerateLongitudeSlice {
            planet_model: planet_model.clone(),
            longitude,
            bounding_plane,
            plane,
            edge_points: [interior_point.clone()],
            interior_point,
            plane_points: [pm.north_pole.clone(), pm.south_pole.clone()],
        })
    }

    /// `GeoDegenerateLongitudeSlice(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let a = read_double(input)?;
        Self::new(planet_model, a)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let distance =
            style.compute_distance_to_plane(pm, &self.plane, x, y, z, &[&self.bounding_plane]);
        let north_distance = style.compute_distance(&pm.north_pole, x, y, z);
        let south_distance = style.compute_distance(&pm.south_pole, x, y, z);
        min(distance, min(north_distance, south_distance))
    }
}

impl SerializableObject for GeoDegenerateLongitudeSlice {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.longitude);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(13)
    }
}

impl_planet_object!(GeoDegenerateLongitudeSlice);
impl_membership_shape!(GeoDegenerateLongitudeSlice);

impl GeoArea for GeoDegenerateLongitudeSlice {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(line_relationship(self, &self.interior_point, path))
    }
}

impl Membership for GeoDegenerateLongitudeSlice {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane.evaluate_is_zero_xyz(x, y, z) && self.bounding_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoDegenerateLongitudeSlice {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_vertical_plane(pm, self.longitude, &self.plane, &[&self.bounding_plane])
            .add_point(&pm.north_pole)
            .add_point(&pm.south_pole);
    }
}

impl GeoShape for GeoDegenerateLongitudeSlice {
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
            &self.plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[&self.bounding_plane],
        )
    }
}

impl GeoAreaShape for GeoDegenerateLongitudeSlice {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.plane, &self.plane_points, &[&self.bounding_plane])
    }
}

impl GeoSizeable for GeoDegenerateLongitudeSlice {
    fn radius(&self) -> f64 {
        PI * 0.5
    }

    fn center(&self) -> GeoPoint {
        self.interior_point.clone()
    }
}

impl GeoBBox for GeoDegenerateLongitudeSlice {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        let mut new_left_lon = self.longitude - angle;
        let mut new_right_lon = self.longitude + angle;
        let current_lon_span = 2.0 * angle;
        if current_lon_span + 2.0 * angle >= PI * 2.0 {
            new_left_lon = -PI;
            new_right_lon = PI;
        }
        make_geo_bbox(
            &self.planet_model,
            PI * 0.5,
            -PI * 0.5,
            new_left_lon,
            new_right_lon,
        )
    }
}

/// A whole latitude.
#[derive(Debug, Clone)]
pub struct GeoDegenerateLatitudeZone {
    planet_model: Arc<PlanetModel>,
    latitude: f64,
    sin_latitude: f64,
    plane: Plane,
    interior_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoDegenerateLatitudeZone {
    /// `GeoDegenerateLatitudeZone(planetModel, latitude)`.
    pub fn new(planet_model: &Arc<PlanetModel>, latitude: f64) -> Result<Self> {
        let pm = &**planet_model;
        let sin_latitude = sin(latitude);
        let cos_latitude = cos(latitude);
        let plane = Plane::horizontal(pm, sin_latitude);
        // Compute an interior point.
        let interior_point = GeoPoint::from_trig(pm, sin_latitude, 0.0, cos_latitude, 1.0);
        Ok(GeoDegenerateLatitudeZone {
            planet_model: planet_model.clone(),
            latitude,
            sin_latitude,
            plane,
            edge_points: [interior_point.clone()],
            interior_point,
        })
    }

    /// `GeoDegenerateLatitudeZone(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let a = read_double(input)?;
        Self::new(planet_model, a)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance_to_plane(&self.planet_model, &self.plane, x, y, z, &[])
    }
}

impl SerializableObject for GeoDegenerateLatitudeZone {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.latitude);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(12)
    }
}

impl_planet_object!(GeoDegenerateLatitudeZone);
impl_membership_shape!(GeoDegenerateLatitudeZone);

impl GeoArea for GeoDegenerateLatitudeZone {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(line_relationship(self, &self.interior_point, path))
    }
}

impl Membership for GeoDegenerateLatitudeZone {
    fn is_within_xyz(&self, _x: f64, _y: f64, z: f64) -> bool {
        abs(z - self.sin_latitude) < 1e-10
    }
}

impl Bounded for GeoDegenerateLatitudeZone {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .no_longitude_bound()
            .add_horizontal_plane(pm, self.latitude, &self.plane, &[]);
    }
}

impl GeoShape for GeoDegenerateLatitudeZone {
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
            &self.plane,
            notable_points,
            &[],
            bounds,
            &[],
        )
    }
}

impl GeoAreaShape for GeoDegenerateLatitudeZone {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.plane, &[], &[])
    }
}

impl GeoSizeable for GeoDegenerateLatitudeZone {
    fn radius(&self) -> f64 {
        PI
    }

    fn center(&self) -> GeoPoint {
        self.interior_point.clone()
    }
}

impl GeoBBox for GeoDegenerateLatitudeZone {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        make_geo_bbox(
            &self.planet_model,
            self.latitude + angle,
            self.latitude - angle,
            -PI,
            PI,
        )
    }
}
