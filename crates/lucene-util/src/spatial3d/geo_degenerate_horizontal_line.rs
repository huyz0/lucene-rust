//! `GeoDegenerateHorizontalLine`
//! (`org.apache.lucene.spatial3d.geom.GeoDegenerateHorizontalLine`) and
//! `GeoWideDegenerateHorizontalLine`: a segment of one latitude, at most
//! half the planet wide, or wider.

#![allow(non_snake_case)]

use super::either_bound::EitherBound;
use super::geo_bbox_factory::make_geo_bbox;
use super::geo_wide_rectangle::MIN_WIDE_EXTENT;
use super::prelude::*;

fn check(latitude: f64, left_lon: f64, right_lon: f64) -> Result<f64> {
    if latitude > PI * 0.5 || latitude < -PI * 0.5 {
        return Err(illegal("Latitude out of range"));
    }
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

fn expand_line(
    pm: &Arc<PlanetModel>,
    latitude: f64,
    left_lon: f64,
    right_lon: f64,
    angle: f64,
) -> Result<Arc<dyn GeoBBox>> {
    let new_top_lat = latitude + angle;
    let new_bottom_lat = latitude - angle;
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
    make_geo_bbox(pm, new_top_lat, new_bottom_lat, new_left_lon, new_right_lon)
}

/// The latitude, its end points, and the center between them.
struct Common {
    LHC: GeoPoint,
    RHC: GeoPoint,
    plane: Plane,
    center_point: GeoPoint,
    sin_left_lon: f64,
    cos_left_lon: f64,
    sin_right_lon: f64,
    cos_right_lon: f64,
}

fn common(pm: &PlanetModel, latitude: f64, left_lon: f64, right_lon: f64) -> Result<Common> {
    let sin_latitude = sin(latitude);
    let cos_latitude = cos(latitude);
    let sin_left_lon = sin(left_lon);
    let cos_left_lon = cos(left_lon);
    let sin_right_lon = sin(right_lon);
    let cos_right_lon = cos(right_lon);
    let LHC = GeoPoint::from_trig_lat_lon(
        pm,
        sin_latitude,
        sin_left_lon,
        cos_latitude,
        cos_left_lon,
        latitude,
        left_lon,
    );
    let LHC = LHC?;
    let RHC = GeoPoint::from_trig_lat_lon(
        pm,
        sin_latitude,
        sin_right_lon,
        cos_latitude,
        cos_right_lon,
        latitude,
        right_lon,
    );
    let RHC = RHC?;
    let plane = Plane::horizontal(pm, sin_latitude);
    let mut rl = right_lon;
    while left_lon > rl {
        rl += PI * 2.0;
    }
    let middle_lon = (left_lon + rl) * 0.5;
    let sin_middle_lon = sin(middle_lon);
    let cos_middle_lon = cos(middle_lon);
    let center_point = GeoPoint::from_trig(
        pm,
        sin_latitude,
        sin_middle_lon,
        cos_latitude,
        cos_middle_lon,
    );
    Ok(Common {
        LHC,
        RHC,
        plane,
        center_point,
        sin_left_lon,
        cos_left_lon,
        sin_right_lon,
        cos_right_lon,
    })
}

/// The overridden `getRelationship(path)` of the degenerate lines: no
/// planet-model check, and no `WITHIN`.
pub(crate) fn line_relationship<S: GeoAreaShape + ?Sized>(
    line: &S,
    center: &GeoPoint,
    path: &dyn GeoShape,
) -> GeoAreaRelationship {
    if line.intersects_shape(path) {
        return GeoAreaRelationship::Overlaps;
    }
    if path.is_within(center) {
        return GeoAreaRelationship::Contains;
    }
    GeoAreaRelationship::Disjoint
}

/// A latitude segment no wider than half the planet.
#[derive(Debug, Clone)]
pub struct GeoDegenerateHorizontalLine {
    planet_model: Arc<PlanetModel>,
    latitude: f64,
    left_lon: f64,
    right_lon: f64,
    LHC: GeoPoint,
    RHC: GeoPoint,
    plane: Plane,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoDegenerateHorizontalLine {
    /// `GeoDegenerateHorizontalLine(planetModel, latitude, leftLon,
    /// rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        latitude: f64,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if check(latitude, left_lon, right_lon)? > PI {
            return Err(illegal("Width of rectangle too great"));
        }
        let c = common(pm, latitude, left_lon, right_lon)?;
        let left_plane = SidedPlane::vertical(&c.RHC, c.cos_left_lon, c.sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&c.LHC, c.cos_right_lon, c.sin_right_lon)?;
        Ok(GeoDegenerateHorizontalLine {
            planet_model: planet_model.clone(),
            latitude,
            left_lon,
            right_lon,
            plane_points: [c.LHC.clone(), c.RHC.clone()],
            edge_points: [c.center_point.clone()],
            LHC: c.LHC,
            RHC: c.RHC,
            plane: c.plane,
            left_plane,
            right_plane,
            center_point: c.center_point,
        })
    }

    /// `GeoDegenerateHorizontalLine(planetModel, InputStream)`.
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
            &[&self.left_plane, &self.right_plane],
        );
        let lhc_distance = style.compute_distance(&self.LHC, x, y, z);
        let rhc_distance = style.compute_distance(&self.RHC, x, y, z);
        min(distance, min(lhc_distance, rhc_distance))
    }
}

impl SerializableObject for GeoDegenerateHorizontalLine {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.latitude);
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(11)
    }
}

impl_planet_object!(GeoDegenerateHorizontalLine);
impl_membership_shape!(GeoDegenerateHorizontalLine);

impl GeoArea for GeoDegenerateHorizontalLine {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(line_relationship(self, &self.center_point, path))
    }
}

impl Membership for GeoDegenerateHorizontalLine {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane.evaluate_is_zero_xyz(x, y, z)
            && self.left_plane.is_within_xyz(x, y, z)
            && self.right_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoDegenerateHorizontalLine {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_horizontal_plane(
                pm,
                self.latitude,
                &self.plane,
                &[&self.left_plane, &self.right_plane],
            )
            .add_point(&self.LHC)
            .add_point(&self.RHC);
    }
}

impl GeoShape for GeoDegenerateHorizontalLine {
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
            &[&self.left_plane, &self.right_plane],
        )
    }
}

impl GeoAreaShape for GeoDegenerateHorizontalLine {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.plane,
            &self.plane_points,
            &[&self.left_plane, &self.right_plane],
        )
    }
}

impl GeoSizeable for GeoDegenerateHorizontalLine {
    fn radius(&self) -> f64 {
        let top_angle = self.center_point.arc_distance(&self.RHC);
        let bottom_angle = self.center_point.arc_distance(&self.LHC);
        max(top_angle, bottom_angle)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoDegenerateHorizontalLine {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        expand_line(
            &self.planet_model,
            self.latitude,
            self.left_lon,
            self.right_lon,
            angle,
        )
    }
}

/// A latitude segment wider than half the planet.
#[derive(Debug, Clone)]
pub struct GeoWideDegenerateHorizontalLine {
    planet_model: Arc<PlanetModel>,
    latitude: f64,
    left_lon: f64,
    right_lon: f64,
    LHC: GeoPoint,
    RHC: GeoPoint,
    plane: Plane,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    either_bound: EitherBound,
    edge_points: [GeoPoint; 1],
}

impl GeoWideDegenerateHorizontalLine {
    /// `GeoWideDegenerateHorizontalLine(planetModel, latitude, leftLon,
    /// rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        latitude: f64,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if check(latitude, left_lon, right_lon)? < MIN_WIDE_EXTENT {
            return Err(illegal("Width of rectangle too small"));
        }
        let c = common(pm, latitude, left_lon, right_lon)?;
        let left_plane = SidedPlane::vertical(&c.center_point, c.cos_left_lon, c.sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&c.center_point, c.cos_right_lon, c.sin_right_lon)?;
        Ok(GeoWideDegenerateHorizontalLine {
            planet_model: planet_model.clone(),
            latitude,
            left_lon,
            right_lon,
            plane_points: [c.LHC.clone(), c.RHC.clone()],
            edge_points: [c.center_point.clone()],
            LHC: c.LHC,
            RHC: c.RHC,
            plane: c.plane,
            left_plane,
            right_plane,
            center_point: c.center_point,
            either_bound: EitherBound {
                left: left_plane,
                right: right_plane,
            },
        })
    }

    /// `GeoWideDegenerateHorizontalLine(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        let c = read_double(input)?;
        Self::new(planet_model, a, b, c)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let distance =
            style.compute_distance_to_plane(pm, &self.plane, x, y, z, &[&self.either_bound]);
        let lhc_distance = style.compute_distance(&self.LHC, x, y, z);
        let rhc_distance = style.compute_distance(&self.RHC, x, y, z);
        min(distance, min(lhc_distance, rhc_distance))
    }
}

impl SerializableObject for GeoWideDegenerateHorizontalLine {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.latitude);
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(21)
    }
}

impl_planet_object!(GeoWideDegenerateHorizontalLine);
impl_membership_shape!(GeoWideDegenerateHorizontalLine);

impl GeoArea for GeoWideDegenerateHorizontalLine {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(line_relationship(self, &self.center_point, path))
    }
}

impl Membership for GeoWideDegenerateHorizontalLine {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane.evaluate_is_zero_xyz(x, y, z)
            && (self.left_plane.is_within_xyz(x, y, z) || self.right_plane.is_within_xyz(x, y, z))
    }
}

impl Bounded for GeoWideDegenerateHorizontalLine {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .is_wide()
            .add_horizontal_plane(pm, self.latitude, &self.plane, &[&self.either_bound])
            .add_point(&self.LHC)
            .add_point(&self.RHC);
    }
}

impl GeoShape for GeoWideDegenerateHorizontalLine {
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
            &[&self.either_bound],
        )
    }
}

impl GeoAreaShape for GeoWideDegenerateHorizontalLine {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(&self.plane, &self.plane_points, &[&self.either_bound])
    }
}

impl GeoSizeable for GeoWideDegenerateHorizontalLine {
    fn radius(&self) -> f64 {
        let top_angle = self.center_point.arc_distance(&self.RHC);
        let bottom_angle = self.center_point.arc_distance(&self.LHC);
        max(top_angle, bottom_angle)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoWideDegenerateHorizontalLine {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        expand_line(
            &self.planet_model,
            self.latitude,
            self.left_lon,
            self.right_lon,
            angle,
        )
    }
}
