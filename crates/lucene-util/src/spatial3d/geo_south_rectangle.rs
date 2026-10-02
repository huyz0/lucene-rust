//! `GeoSouthRectangle` (`org.apache.lucene.spatial3d.geom.GeoSouthRectangle`):
//! a box that reaches the south pole, no wider than half the planet.

#![allow(non_snake_case)]

use super::geo_bbox_factory::make_geo_bbox;
use super::prelude::*;

/// A box with its bottom at the south pole.
#[derive(Debug, Clone)]
pub struct GeoSouthRectangle {
    planet_model: Arc<PlanetModel>,
    top_lat: f64,
    left_lon: f64,
    right_lon: f64,
    cos_middle_lat: f64,
    URHC: GeoPoint,
    ULHC: GeoPoint,
    top_plane: SidedPlane,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    backing_plane: SidedPlane,
    top_plane_points: [GeoPoint; 2],
    left_plane_points: [GeoPoint; 2],
    right_plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    edge_points: [GeoPoint; 1],
}

impl GeoSouthRectangle {
    /// `GeoSouthRectangle(planetModel, topLat, leftLon, rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        top_lat: f64,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<GeoSouthRectangle> {
        let pm = &**planet_model;
        if top_lat > PI * 0.5 || top_lat < -PI * 0.5 {
            return Err(illegal("Top latitude out of range"));
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
        if extent > PI {
            return Err(illegal("Width of rectangle too great"));
        }
        let sin_top_lat = sin(top_lat);
        let cos_top_lat = cos(top_lat);
        let sin_left_lon = sin(left_lon);
        let cos_left_lon = cos(left_lon);
        let sin_right_lon = sin(right_lon);
        let cos_right_lon = cos(right_lon);
        let URHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_top_lat,
            sin_right_lon,
            cos_top_lat,
            cos_right_lon,
            top_lat,
            right_lon,
        );
        let URHC = URHC?;
        let ULHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_top_lat,
            sin_left_lon,
            cos_top_lat,
            cos_left_lon,
            top_lat,
            left_lon,
        );
        let ULHC = ULHC?;
        let middle_lat = (top_lat - PI * 0.5) * 0.5;
        let sin_middle_lat = sin(middle_lat);
        let cos_middle_lat = cos(middle_lat);
        let mut rl = right_lon;
        while left_lon > rl {
            rl += PI * 2.0;
        }
        let middle_lon = (left_lon + rl) * 0.5;
        let sin_middle_lon = sin(middle_lon);
        let cos_middle_lon = cos(middle_lon);
        let center_point = GeoPoint::from_trig(
            pm,
            sin_middle_lat,
            sin_middle_lon,
            cos_middle_lat,
            cos_middle_lon,
        );
        let top_plane = SidedPlane::horizontal(&pm.south_pole, pm, sin_top_lat)?;
        let left_plane = SidedPlane::vertical(&URHC, cos_left_lon, sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&ULHC, cos_right_lon, sin_right_lon)?;
        let backing_plane =
            SidedPlane::from_abcd(&center_point, cos_middle_lon, sin_middle_lon, 0.0, 0.0)?;
        Ok(GeoSouthRectangle {
            planet_model: planet_model.clone(),
            top_lat,
            left_lon,
            right_lon,
            cos_middle_lat,
            top_plane_points: [ULHC.clone(), URHC.clone()],
            left_plane_points: [ULHC.clone(), pm.south_pole.clone()],
            right_plane_points: [URHC.clone(), pm.south_pole.clone()],
            edge_points: [pm.south_pole.clone()],
            URHC,
            ULHC,
            top_plane,
            left_plane,
            right_plane,
            backing_plane,
            center_point,
        })
    }

    /// `GeoSouthRectangle(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoSouthRectangle> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        let c = read_double(input)?;
        GeoSouthRectangle::new(planet_model, a, b, c)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let top_distance = style.compute_distance_to_plane(
            pm,
            &self.top_plane,
            x,
            y,
            z,
            &[&self.left_plane, &self.right_plane],
        );
        let left_distance = style.compute_distance_to_plane(
            pm,
            &self.left_plane,
            x,
            y,
            z,
            &[&self.right_plane, &self.top_plane],
        );
        let right_distance = style.compute_distance_to_plane(
            pm,
            &self.right_plane,
            x,
            y,
            z,
            &[&self.left_plane, &self.top_plane],
        );
        let ulhc_distance = style.compute_distance(&self.ULHC, x, y, z);
        let urhc_distance = style.compute_distance(&self.URHC, x, y, z);
        min(
            min(top_distance, min(left_distance, right_distance)),
            min(ulhc_distance, urhc_distance),
        )
    }
}

impl SerializableObject for GeoSouthRectangle {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.top_lat);
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(20)
    }
}

impl_planet_object!(GeoSouthRectangle);
impl_membership_shape!(GeoSouthRectangle);
impl_base_area!(GeoSouthRectangle);

impl Membership for GeoSouthRectangle {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.backing_plane.is_within_xyz(x, y, z)
            && self.top_plane.is_within_xyz(x, y, z)
            && self.left_plane.is_within_xyz(x, y, z)
            && self.right_plane.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoSouthRectangle {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_horizontal_plane(
                pm,
                self.top_lat,
                &self.top_plane,
                &[&self.left_plane, &self.right_plane],
            )
            .add_vertical_plane(
                pm,
                self.left_lon,
                &self.left_plane,
                &[&self.top_plane, &self.right_plane],
            )
            .add_vertical_plane(
                pm,
                self.right_lon,
                &self.right_plane,
                &[&self.top_plane, &self.left_plane],
            )
            .add_point(&self.URHC)
            .add_point(&self.ULHC)
            .add_point(&pm.south_pole);
    }
}

impl GeoShape for GeoSouthRectangle {
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
            &self.top_plane_points,
            bounds,
            &[&self.left_plane, &self.right_plane],
        ) || p.intersects(
            pm,
            &self.left_plane,
            notable_points,
            &self.left_plane_points,
            bounds,
            &[&self.right_plane, &self.top_plane],
        ) || p.intersects(
            pm,
            &self.right_plane,
            notable_points,
            &self.right_plane_points,
            bounds,
            &[&self.left_plane, &self.top_plane],
        )
    }
}

impl GeoAreaShape for GeoSouthRectangle {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.top_plane,
            &self.top_plane_points,
            &[&self.left_plane, &self.right_plane],
        ) || geo_shape.intersects(
            &self.left_plane,
            &self.left_plane_points,
            &[&self.right_plane, &self.top_plane],
        ) || geo_shape.intersects(
            &self.right_plane,
            &self.right_plane_points,
            &[&self.left_plane, &self.top_plane],
        )
    }
}

impl GeoSizeable for GeoSouthRectangle {
    fn radius(&self) -> f64 {
        let center_angle =
            (self.right_lon - (self.right_lon + self.left_lon) * 0.5) * self.cos_middle_lat;
        let top_angle = self.center_point.arc_distance(&self.URHC);
        max(center_angle, top_angle)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoSouthRectangle {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        let new_top_lat = self.top_lat + angle;
        let new_bottom_lat = -PI * 0.5;
        let mut current_lon_span = self.right_lon - self.left_lon;
        if current_lon_span < 0.0 {
            current_lon_span += PI * 2.0;
        }
        let mut new_left_lon = self.left_lon - angle;
        let mut new_right_lon = self.right_lon + angle;
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
