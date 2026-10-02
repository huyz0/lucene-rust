//! `GeoLongitudeSlice` (`org.apache.lucene.spatial3d.geom.GeoLongitudeSlice`)
//! and `GeoWideLongitudeSlice`: every latitude between two longitudes, pole
//! to pole -- at most half the planet wide, or wider.

use super::geo_bbox_factory::make_geo_bbox;
use super::geo_wide_rectangle::MIN_WIDE_EXTENT;
use super::prelude::*;

fn check_lons(left_lon: f64, right_lon: f64) -> Result<f64> {
    if left_lon < -PI || left_lon > PI {
        return Err(illegal("Left longitude out of range"));
    }
    if right_lon < -PI || right_lon > PI {
        return Err(illegal("Right longitude out of range"));
    }
    let mut extent = right_lon - left_lon;
    if extent < 0.0 {
        extent += 2.0 * PI;
    }
    Ok(extent)
}

/// The slices' shared `expand`.
fn expand_slice(
    pm: &Arc<PlanetModel>,
    left_lon: f64,
    right_lon: f64,
    angle: f64,
) -> Result<Arc<dyn GeoBBox>> {
    let mut current_lon_span = right_lon - left_lon;
    if current_lon_span < 0.0 {
        current_lon_span += PI * 2.0;
    }
    let mut new_left_lon = left_lon - angle;
    let mut new_right_lon = right_lon + angle;
    if current_lon_span + 2.0 * angle >= PI * 2.0 {
        new_left_lon = -PI;
        new_right_lon = PI;
    }
    make_geo_bbox(pm, PI * 0.5, -PI * 0.5, new_left_lon, new_right_lon)
}

/// The slices' shared `getRadius`.
fn slice_radius(left_lon: f64, right_lon: f64) -> f64 {
    let mut extent = right_lon - left_lon;
    if extent < 0.0 {
        extent += PI * 2.0;
    }
    max(PI * 0.5, extent * 0.5)
}

/// A slice no wider than half the planet.
#[derive(Debug, Clone)]
pub struct GeoLongitudeSlice {
    planet_model: Arc<PlanetModel>,
    left_lon: f64,
    right_lon: f64,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    backing_plane: SidedPlane,
    plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoLongitudeSlice {
    /// `GeoLongitudeSlice(planetModel, leftLon, rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<GeoLongitudeSlice> {
        let pm = &**planet_model;
        let extent = check_lons(left_lon, right_lon)?;
        if extent > PI {
            return Err(illegal("Width of rectangle too great"));
        }
        let sin_left_lon = sin(left_lon);
        let cos_left_lon = cos(left_lon);
        let sin_right_lon = sin(right_lon);
        let cos_right_lon = cos(right_lon);
        let mut rl = right_lon;
        while left_lon > rl {
            rl += PI * 2.0;
        }
        let middle_lon = (left_lon + rl) * 0.5;
        let sin_middle_lon = sin(middle_lon);
        let cos_middle_lon = cos(middle_lon);
        let center_point = GeoPoint::from_trig(pm, 0.0, sin_middle_lon, 1.0, cos_middle_lon);
        let left_plane = SidedPlane::vertical(&center_point, cos_left_lon, sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&center_point, cos_right_lon, sin_right_lon)?;
        let backing_plane =
            SidedPlane::from_abcd(&center_point, cos_middle_lon, sin_middle_lon, 0.0, 0.0)?;
        Ok(GeoLongitudeSlice {
            planet_model: planet_model.clone(),
            left_lon,
            right_lon,
            left_plane,
            right_plane,
            backing_plane,
            plane_points: [pm.north_pole.clone(), pm.south_pole.clone()],
            center_point,
            edge_points: [pm.north_pole.clone()],
        })
    }

    /// `GeoLongitudeSlice(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoLongitudeSlice> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        GeoLongitudeSlice::new(planet_model, a, b)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let left_distance =
            style.compute_distance_to_plane(pm, &self.left_plane, x, y, z, &[&self.right_plane]);
        let right_distance =
            style.compute_distance_to_plane(pm, &self.right_plane, x, y, z, &[&self.left_plane]);
        let north_distance = style.compute_distance(&pm.north_pole, x, y, z);
        let south_distance = style.compute_distance(&pm.south_pole, x, y, z);
        min(
            min(north_distance, south_distance),
            min(left_distance, right_distance),
        )
    }
}

impl SerializableObject for GeoLongitudeSlice {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(16)
    }
}

impl_planet_object!(GeoLongitudeSlice);
impl_membership_shape!(GeoLongitudeSlice);
impl_base_area!(GeoLongitudeSlice);

impl Membership for GeoLongitudeSlice {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.backing_plane.is_within_xyz(x, y, z)
            && self.left_plane.is_within_xyz(x, y, z)
            && self.right_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoLongitudeSlice {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_vertical_plane(pm, self.left_lon, &self.left_plane, &[&self.right_plane])
            .add_vertical_plane(pm, self.right_lon, &self.right_plane, &[&self.left_plane])
            .add_point(&pm.north_pole)
            .add_point(&pm.south_pole);
    }
}

impl GeoShape for GeoLongitudeSlice {
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
            &self.left_plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[&self.right_plane],
        ) || p.intersects(
            pm,
            &self.right_plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[&self.left_plane],
        )
    }
}

impl GeoAreaShape for GeoLongitudeSlice {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.left_plane, &self.plane_points, &[&self.right_plane])
            || geo_shape.intersects(&self.right_plane, &self.plane_points, &[&self.left_plane])
    }
}

impl GeoSizeable for GeoLongitudeSlice {
    fn radius(&self) -> f64 {
        slice_radius(self.left_lon, self.right_lon)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoLongitudeSlice {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        expand_slice(&self.planet_model, self.left_lon, self.right_lon, angle)
    }
}

/// A slice wider than half the planet.
#[derive(Debug, Clone)]
pub struct GeoWideLongitudeSlice {
    planet_model: Arc<PlanetModel>,
    left_lon: f64,
    right_lon: f64,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoWideLongitudeSlice {
    /// `GeoWideLongitudeSlice(planetModel, leftLon, rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<GeoWideLongitudeSlice> {
        let pm = &**planet_model;
        let extent = check_lons(left_lon, right_lon)?;
        if extent < MIN_WIDE_EXTENT {
            return Err(illegal("Width of rectangle too small"));
        }
        let sin_left_lon = sin(left_lon);
        let cos_left_lon = cos(left_lon);
        let sin_right_lon = sin(right_lon);
        let cos_right_lon = cos(right_lon);
        let mut rl = right_lon;
        while left_lon > rl {
            rl += PI * 2.0;
        }
        let mut middle_lon = (left_lon + rl) * 0.5;
        while middle_lon > PI {
            middle_lon -= PI * 2.0;
        }
        while middle_lon < -PI {
            middle_lon += PI * 2.0;
        }
        let center_point = GeoPoint::from_lat_lon(pm, 0.0, middle_lon)?;
        let left_plane = SidedPlane::vertical(&center_point, cos_left_lon, sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&center_point, cos_right_lon, sin_right_lon)?;
        Ok(GeoWideLongitudeSlice {
            planet_model: planet_model.clone(),
            left_lon,
            right_lon,
            left_plane,
            right_plane,
            plane_points: [pm.north_pole.clone(), pm.south_pole.clone()],
            center_point,
            edge_points: [pm.north_pole.clone()],
        })
    }

    /// `GeoWideLongitudeSlice(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoWideLongitudeSlice> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        GeoWideLongitudeSlice::new(planet_model, a, b)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let left_distance = style.compute_distance_to_plane(pm, &self.left_plane, x, y, z, &[]);
        let right_distance = style.compute_distance_to_plane(pm, &self.right_plane, x, y, z, &[]);
        let north_distance = style.compute_distance(&pm.north_pole, x, y, z);
        let south_distance = style.compute_distance(&pm.south_pole, x, y, z);
        min(
            min(left_distance, right_distance),
            min(north_distance, south_distance),
        )
    }
}

impl SerializableObject for GeoWideLongitudeSlice {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(22)
    }
}

impl_planet_object!(GeoWideLongitudeSlice);
impl_membership_shape!(GeoWideLongitudeSlice);
impl_base_area!(GeoWideLongitudeSlice);

impl Membership for GeoWideLongitudeSlice {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.left_plane.is_within_xyz(x, y, z) || self.right_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoWideLongitudeSlice {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .is_wide()
            .add_vertical_plane(pm, self.left_lon, &self.left_plane, &[])
            .add_vertical_plane(pm, self.right_lon, &self.right_plane, &[])
            .add_intersection(pm, &self.left_plane, &self.right_plane, &[])
            .add_point(&pm.north_pole)
            .add_point(&pm.south_pole);
    }
}

impl GeoShape for GeoWideLongitudeSlice {
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
            &self.left_plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[],
        ) || p.intersects(
            pm,
            &self.right_plane,
            notable_points,
            &self.plane_points,
            bounds,
            &[],
        )
    }
}

impl GeoAreaShape for GeoWideLongitudeSlice {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.left_plane, &self.plane_points, &[])
            || geo_shape.intersects(&self.right_plane, &self.plane_points, &[])
    }
}

impl GeoSizeable for GeoWideLongitudeSlice {
    fn radius(&self) -> f64 {
        slice_radius(self.left_lon, self.right_lon)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoWideLongitudeSlice {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        expand_slice(&self.planet_model, self.left_lon, self.right_lon, angle)
    }
}
