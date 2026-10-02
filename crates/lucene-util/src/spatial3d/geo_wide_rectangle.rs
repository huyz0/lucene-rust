//! `GeoWideRectangle` (`org.apache.lucene.spatial3d.geom.GeoWideRectangle`):
//! a latitude/longitude box wider than half the planet, not touching a
//! pole.

#![allow(non_snake_case)]

use super::either_bound::EitherBound;
use super::geo_bbox_factory::make_geo_bbox;
use super::prelude::*;

/// `GeoWideRectangle.MIN_WIDE_EXTENT`.
pub const MIN_WIDE_EXTENT: f64 = PI - MINIMUM_ANGULAR_RESOLUTION;

/// A wide bounding box.
#[derive(Debug, Clone)]
pub struct GeoWideRectangle {
    planet_model: Arc<PlanetModel>,
    top_lat: f64,
    bottom_lat: f64,
    left_lon: f64,
    right_lon: f64,
    cos_middle_lat: f64,
    ULHC: GeoPoint,
    URHC: GeoPoint,
    LRHC: GeoPoint,
    LLHC: GeoPoint,
    top_plane: SidedPlane,
    bottom_plane: SidedPlane,
    left_plane: SidedPlane,
    right_plane: SidedPlane,
    top_plane_points: [GeoPoint; 2],
    bottom_plane_points: [GeoPoint; 2],
    left_plane_points: [GeoPoint; 2],
    right_plane_points: [GeoPoint; 2],
    center_point: GeoPoint,
    either_bound: EitherBound,
    edge_points: [GeoPoint; 1],
}

impl GeoWideRectangle {
    /// `GeoWideRectangle(planetModel, topLat, bottomLat, leftLon, rightLon)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        top_lat: f64,
        bottom_lat: f64,
        left_lon: f64,
        right_lon: f64,
    ) -> Result<GeoWideRectangle> {
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
        let sin_top_lat = sin(top_lat);
        let cos_top_lat = cos(top_lat);
        let sin_bottom_lat = sin(bottom_lat);
        let cos_bottom_lat = cos(bottom_lat);
        let sin_left_lon = sin(left_lon);
        let cos_left_lon = cos(left_lon);
        let sin_right_lon = sin(right_lon);
        let cos_right_lon = cos(right_lon);
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
        let middle_lat = (top_lat + bottom_lat) * 0.5;
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
        let top_plane = SidedPlane::horizontal(&center_point, pm, sin_top_lat)?;
        let bottom_plane = SidedPlane::horizontal(&center_point, pm, sin_bottom_lat)?;
        let left_plane = SidedPlane::vertical(&center_point, cos_left_lon, sin_left_lon)?;
        let right_plane = SidedPlane::vertical(&center_point, cos_right_lon, sin_right_lon)?;
        Ok(GeoWideRectangle {
            planet_model: planet_model.clone(),
            top_lat,
            bottom_lat,
            left_lon,
            right_lon,
            cos_middle_lat,
            top_plane_points: [ULHC.clone(), URHC.clone()],
            bottom_plane_points: [LLHC.clone(), LRHC.clone()],
            left_plane_points: [ULHC.clone(), LLHC.clone()],
            right_plane_points: [URHC.clone(), LRHC.clone()],
            edge_points: [ULHC.clone()],
            ULHC,
            URHC,
            LRHC,
            LLHC,
            top_plane,
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

    /// `GeoWideRectangle(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoWideRectangle> {
        let a = read_double(input)?;
        let b = read_double(input)?;
        let c = read_double(input)?;
        let d = read_double(input)?;
        GeoWideRectangle::new(planet_model, a, b, c, d)
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let top_distance = style.compute_distance_to_plane(
            pm,
            &self.top_plane,
            x,
            y,
            z,
            &[&self.bottom_plane, &self.either_bound],
        );
        let bottom_distance = style.compute_distance_to_plane(
            pm,
            &self.bottom_plane,
            x,
            y,
            z,
            &[&self.top_plane, &self.either_bound],
        );
        let left_distance = style.compute_distance_to_plane(
            pm,
            &self.left_plane,
            x,
            y,
            z,
            &[&self.top_plane, &self.bottom_plane],
        );
        let right_distance = style.compute_distance_to_plane(
            pm,
            &self.right_plane,
            x,
            y,
            z,
            &[&self.top_plane, &self.bottom_plane],
        );
        let ulhc_distance = style.compute_distance(&self.ULHC, x, y, z);
        let urhc_distance = style.compute_distance(&self.URHC, x, y, z);
        let lrhc_distance = style.compute_distance(&self.LRHC, x, y, z);
        let llhc_distance = style.compute_distance(&self.LLHC, x, y, z);
        min(
            min(
                min(top_distance, bottom_distance),
                min(left_distance, right_distance),
            ),
            min(
                min(ulhc_distance, urhc_distance),
                min(lrhc_distance, llhc_distance),
            ),
        )
    }
}

impl SerializableObject for GeoWideRectangle {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.top_lat);
        write_double(out, self.bottom_lat);
        write_double(out, self.left_lon);
        write_double(out, self.right_lon);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(24)
    }
}

impl_planet_object!(GeoWideRectangle);
impl_membership_shape!(GeoWideRectangle);
impl_base_area!(GeoWideRectangle);

impl Membership for GeoWideRectangle {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.top_plane.is_within_xyz(x, y, z)
            && self.bottom_plane.is_within_xyz(x, y, z)
            && (self.left_plane.is_within_xyz(x, y, z) || self.right_plane.is_within_xyz(x, y, z))
    }
}

impl Bounded for GeoWideRectangle {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .is_wide()
            .add_horizontal_plane(
                pm,
                self.top_lat,
                &self.top_plane,
                &[&self.bottom_plane, &self.either_bound],
            )
            .add_vertical_plane(
                pm,
                self.right_lon,
                &self.right_plane,
                &[&self.top_plane, &self.bottom_plane],
            )
            .add_horizontal_plane(
                pm,
                self.bottom_lat,
                &self.bottom_plane,
                &[&self.top_plane, &self.either_bound],
            )
            .add_vertical_plane(
                pm,
                self.left_lon,
                &self.left_plane,
                &[&self.top_plane, &self.bottom_plane],
            )
            .add_intersection(
                pm,
                &self.left_plane,
                &self.right_plane,
                &[&self.top_plane, &self.bottom_plane],
            )
            .add_point(&self.ULHC)
            .add_point(&self.URHC)
            .add_point(&self.LRHC)
            .add_point(&self.LLHC);
    }
}

impl GeoShape for GeoWideRectangle {
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
            &[&self.bottom_plane, &self.either_bound],
        ) || p.intersects(
            pm,
            &self.bottom_plane,
            notable_points,
            &self.bottom_plane_points,
            bounds,
            &[&self.top_plane, &self.either_bound],
        ) || p.intersects(
            pm,
            &self.left_plane,
            notable_points,
            &self.left_plane_points,
            bounds,
            &[&self.top_plane, &self.bottom_plane],
        ) || p.intersects(
            pm,
            &self.right_plane,
            notable_points,
            &self.right_plane_points,
            bounds,
            &[&self.top_plane, &self.bottom_plane],
        )
    }
}

impl GeoAreaShape for GeoWideRectangle {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.top_plane,
            &self.top_plane_points,
            &[&self.bottom_plane, &self.either_bound],
        ) || geo_shape.intersects(
            &self.bottom_plane,
            &self.bottom_plane_points,
            &[&self.top_plane, &self.either_bound],
        ) || geo_shape.intersects(
            &self.left_plane,
            &self.left_plane_points,
            &[&self.top_plane, &self.bottom_plane],
        ) || geo_shape.intersects(
            &self.right_plane,
            &self.right_plane_points,
            &[&self.top_plane, &self.bottom_plane],
        )
    }
}

impl GeoSizeable for GeoWideRectangle {
    fn radius(&self) -> f64 {
        let center_angle =
            (self.right_lon - (self.right_lon + self.left_lon) * 0.5) * self.cos_middle_lat;
        let top_angle = self.center_point.arc_distance(&self.URHC);
        let bottom_angle = self.center_point.arc_distance(&self.LLHC);
        max(center_angle, max(top_angle, bottom_angle))
    }

    fn center(&self) -> GeoPoint {
        self.center_point.clone()
    }
}

impl GeoBBox for GeoWideRectangle {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        let new_top_lat = self.top_lat + angle;
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
