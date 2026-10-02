//! `GeoWideNorthRectangle`
//! (`org.apache.lucene.spatial3d.geom.GeoWideNorthRectangle`): a box that
//! reaches the north pole, wider than half the planet.

#![allow(non_snake_case)]

use super::either_bound::EitherBound;
use super::geo_bbox_factory::make_geo_bbox;
use super::geo_wide_rectangle::MIN_WIDE_EXTENT;
use super::prelude::*;

/// A wide box with its top at the north pole.
#[derive(Debug, Clone)]
pub struct GeoWideNorthRectangle {
    planet_model: Arc<PlanetModel>,
    bottom_lat: f64,
    left_lon: f64,
    right_lon: f64,
    cos_middle_lat: f64,
    LRHC: GeoPoint,
    LLHC: GeoPoint,
    bottom_plane: SidedPlane,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    bottom_plane_points: [GeoPoint; 2],
    left_plane_points: [GeoPoint; 2],
    right_plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    either_bound: EitherBound,
    edge_points: [GeoPoint; 1],
}

impl GeoWideNorthRectangle {
    /// `GeoWideNorthRectangle(planetModel, bottomLat, leftLon, rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        bottom_lat: f64,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<GeoWideNorthRectangle> {
        let pm = &**planet_model;
        if bottom_lat > PI * 0.5 || bottom_lat < -PI * 0.5 {
            return Err(illegal("Bottom latitude out of range"));
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
        if extent < MIN_WIDE_EXTENT {
            return Err(illegal("Width of rectangle too small"));
        }
        let sin_bottom_lat = sin(bottom_lat);
        let cos_bottom_lat = cos(bottom_lat);
        let sin_left_lon = sin(left_lon);
        let cos_left_lon = cos(left_lon);
        let sin_right_lon = sin(right_lon);
        let cos_right_lon = cos(right_lon);
        let LRHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_bottom_lat,
            sin_right_lon,
            cos_bottom_lat,
            cos_right_lon,
            bottom_lat,
            right_lon,
        );
        let LRHC = LRHC?;
        let LLHC = GeoPoint::from_trig_lat_lon(
            pm,
            sin_bottom_lat,
            sin_left_lon,
            cos_bottom_lat,
            cos_left_lon,
            bottom_lat,
            left_lon,
        );
        let LLHC = LLHC?;
        let middle_lat = (PI * 0.5 + bottom_lat) * 0.5;
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
        let bottom_plane = SidedPlane::horizontal(&center_point, pm, sin_bottom_lat)?;
        let left_plane = SidedPlane::vertical(&center_point, cos_left_lon, sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&center_point, cos_right_lon, sin_right_lon)?;
        Ok(GeoWideNorthRectangle {
            planet_model: planet_model.clone(),
            bottom_lat,
            left_lon,
            right_lon,
            cos_middle_lat,
            bottom_plane_points: [LLHC.clone(), LRHC.clone()],
            left_plane_points: [pm.north_pole.clone(), LLHC.clone()],
            right_plane_points: [pm.north_pole.clone(), LRHC.clone()],
            edge_points: [pm.north_pole.clone()],
            LRHC,
            LLHC,
            bottom_plane,
            left_plane,
            right_plane,
            center_point,
            either_bound: EitherBound {
                left: left_plane,
                right: right_plane,
            },
        })
    }

    /// `GeoWideNorthRectangle(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoWideNorthRectangle> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        let c = read_double(input)?;
        GeoWideNorthRectangle::new(planet_model, a, b, c)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let bottom_distance =
            style.compute_distance_to_plane(pm, &self.bottom_plane, x, y, z, &[&self.either_bound]);
        let left_distance =
            style.compute_distance_to_plane(pm, &self.left_plane, x, y, z, &[&self.bottom_plane]);
        let right_distance =
            style.compute_distance_to_plane(pm, &self.right_plane, x, y, z, &[&self.bottom_plane]);
        let lrhc_distance = style.compute_distance(&self.LRHC, x, y, z);
        let llhc_distance = style.compute_distance(&self.LLHC, x, y, z);
        min(
            min(bottom_distance, min(left_distance, right_distance)),
            min(lrhc_distance, llhc_distance),
        )
    }
}

impl SerializableObject for GeoWideNorthRectangle {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.bottom_lat);
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(23)
    }
}

impl_planet_object!(GeoWideNorthRectangle);
impl_membership_shape!(GeoWideNorthRectangle);
impl_base_area!(GeoWideNorthRectangle);

impl Membership for GeoWideNorthRectangle {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.bottom_plane.is_within_xyz(x, y, z)
            && (self.left_plane.is_within_xyz(x, y, z) || self.right_plane.is_within_xyz(x, y, z))
    }
}

impl Bounded for GeoWideNorthRectangle {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .is_wide()
            .add_horizontal_plane(
                pm,
                self.bottom_lat,
                &self.bottom_plane,
                &[&self.either_bound],
            )
            .add_vertical_plane(pm, self.left_lon, &self.left_plane, &[&self.bottom_plane])
            .add_vertical_plane(pm, self.right_lon, &self.right_plane, &[&self.bottom_plane])
            .add_intersection(
                pm,
                &self.left_plane,
                &self.right_plane,
                &[&self.bottom_plane],
            )
            .add_point(&self.LLHC)
            .add_point(&self.LRHC)
            .add_point(&pm.north_pole);
    }
}

impl GeoShape for GeoWideNorthRectangle {
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
            &self.bottom_plane,
            notable_points,
            &self.bottom_plane_points,
            bounds,
            &[&self.either_bound],
        ) || p.intersects(
            pm,
            &self.left_plane,
            notable_points,
            &self.left_plane_points,
            bounds,
            &[&self.bottom_plane],
        ) || p.intersects(
            pm,
            &self.right_plane,
            notable_points,
            &self.right_plane_points,
            bounds,
            &[&self.bottom_plane],
        )
    }
}

impl GeoAreaShape for GeoWideNorthRectangle {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.bottom_plane,
            &self.bottom_plane_points,
            &[&self.either_bound],
        ) || geo_shape.intersects(
            &self.left_plane,
            &self.left_plane_points,
            &[&self.bottom_plane],
        ) || geo_shape.intersects(
            &self.right_plane,
            &self.right_plane_points,
            &[&self.bottom_plane],
        )
    }
}

impl GeoSizeable for GeoWideNorthRectangle {
    fn radius(&self) -> f64 {
        let center_angle =
            (self.right_lon - (self.right_lon + self.left_lon) * 0.5) * self.cos_middle_lat;
        let bottom_angle = self.center_point.arc_distance(&self.LLHC);
        max(center_angle, bottom_angle)
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoWideNorthRectangle {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        let new_top_lat = PI * 0.5;
        let new_bottom_lat = self.bottom_lat - angle;
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
