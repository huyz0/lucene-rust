//! `GeoStandardCircle` (`org.apache.lucene.spatial3d.geom.GeoStandardCircle`):
//! a circle approximated by one plane cutting the ellipsoid.

use super::prelude::*;
use super::shape::{base_get_relationship, impl_distance_shape, GeoCircle};

/// `GeoStandardCircle.circlePoints`: none.
const CIRCLE_POINTS: &[GeoPoint] = &[];

/// A circle: everything on the center's side of one plane.
#[derive(Debug, Clone)]
pub struct GeoStandardCircle {
    planet_model: Arc<PlanetModel>,
    center: GeoPoint,
    cutoff_angle: f64,
    /// `None` for a circle of radius pi: the whole world.
    circle_plane: Option<SidedPlane>,
    edge_points: Vec<GeoPoint>,
}

impl GeoStandardCircle {
    /// `GeoStandardCircle(planetModel, lat, lon, cutoffAngle)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        lat: f64,
        lon: f64,
        cutoff_angle: f64,
    ) -> Result<GeoStandardCircle> {
        let pm = &**planet_model;
        if lat < -PI * 0.5 || lat > PI * 0.5 {
            return Err(illegal("Latitude out of bounds"));
        }
        if lon < -PI || lon > PI {
            return Err(illegal("Longitude out of bounds"));
        }
        if cutoff_angle < 0.0 || cutoff_angle > PI {
            return Err(illegal("Cutoff angle out of bounds"));
        }
        if cutoff_angle < MINIMUM_RESOLUTION {
            return Err(illegal("Cutoff angle cannot be effectively zero"));
        }
        let center = GeoPoint::from_lat_lon(pm, lat, lon)?;
        // In an ellipsoidal world, cutoff distances make no sense, unfortunately.
        // Only membership can be used to make in/out determination.
        // Compute two points on the circle, with the right angle from the center.
        // We'll use these to obtain the perpendicular plane to the circle.
        let mut upper_lat = lat + cutoff_angle;
        let mut upper_lon = lon;
        if upper_lat > PI * 0.5 {
            upper_lon += PI;
            if upper_lon > PI {
                upper_lon -= 2.0 * PI;
            }
            upper_lat = PI - upper_lat;
        }
        let mut lower_lat = lat - cutoff_angle;
        let mut lower_lon = lon;
        if lower_lat < -PI * 0.5 {
            lower_lon += PI;
            if lower_lon > PI {
                lower_lon -= 2.0 * PI;
            }
            lower_lat = -PI - lower_lat;
        }
        let upper_point = GeoPoint::from_lat_lon(pm, upper_lat, upper_lon)?;
        let lower_point = GeoPoint::from_lat_lon(pm, lower_lat, lower_lon)?;
        let (circle_plane, edge_points) = if abs(cutoff_angle - PI) < MINIMUM_RESOLUTION {
            // Circle is the whole world
            (None, Vec::new())
        } else {
            // Construct normal plane
            let normal_plane =
                Plane::construct_normalized_z_plane_points(&[&upper_point, &lower_point, &center])
                    .ok_or_else(|| Error::NullPointer("all points are on the z axis".into()))?;
            // Construct a sided plane that goes through the two points and whose
            // normal is in the normalPlane.
            let Some(circle_plane) = SidedPlane::construct_normalized_perpendicular_sided_plane(
                &center,
                &normal_plane,
                &upper_point,
                &lower_point,
            )?
            else {
                use crate::geo::java_double_string as d;
                return Err(Error::IllegalArgument(format!(
                    "Couldn't construct circle plane, probably too small?  Cutoff angle = {}; upperPoint = {}; lowerPoint = {}",
                    d(cutoff_angle),
                    upper_point,
                    lower_point
                )));
            };
            let Some(recomputed_intersection_point) =
                circle_plane.sample_intersection_point(pm, &normal_plane)
            else {
                return Err(Error::IllegalArgument(format!(
                    "Couldn't construct intersection point, probably circle too small?  Plane = {circle_plane}"
                )));
            };
            (Some(circle_plane), vec![recomputed_intersection_point])
        };
        Ok(GeoStandardCircle {
            planet_model: planet_model.clone(),
            center,
            cutoff_angle,
            circle_plane,
            edge_points,
        })
    }

    /// `GeoStandardCircle(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoStandardCircle> {
        let lat = read_double(input)?;
        let lon = read_double(input)?;
        let cutoff = read_double(input)?;
        GeoStandardCircle::new(planet_model, lat, lon, cutoff)
    }

    fn distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance(&self.center, x, y, z)
    }

    fn delta_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        self.distance(style, x, y, z) * 2.0
    }

    fn distance_bounds(
        &self,
        bounds: &mut dyn Bounds,
        _style: DistanceStyle,
        _distance_value: f64,
    ) -> Result<()> {
        // TBD: Compute actual bounds based on distance
        self.get_bounds(bounds);
        Ok(())
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        // Only reached outside the circle, so the plane is there (a
        // whole-world circle has none, and nothing outside it).
        self.circle_plane.as_ref().map_or(0.0, |p| {
            style.compute_distance_to_plane(&self.planet_model, p, x, y, z, &[])
        })
    }
}

impl SerializableObject for GeoStandardCircle {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.center.latitude());
        write_double(out, self.center.longitude());
        write_double(out, self.cutoff_angle);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(2)
    }
}

impl_planet_object!(GeoStandardCircle);
impl_membership_shape!(GeoStandardCircle);
impl_distance_shape!(GeoStandardCircle);

impl GeoArea for GeoStandardCircle {
    fn get_relationship(&self, geo_shape: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        if self.circle_plane.is_none() {
            // same as GeoWorld
            if !geo_shape.edge_points().is_empty() {
                return Ok(GeoAreaRelationship::Within);
            }
            return Ok(GeoAreaRelationship::Overlaps);
        }
        base_get_relationship(self, geo_shape)
    }
}

impl Membership for GeoStandardCircle {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        match &self.circle_plane {
            None => true,
            Some(p) => p.is_within_xyz(x, y, z),
        }
    }
}

impl Bounded for GeoStandardCircle {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        let Some(p) = &self.circle_plane else {
            return;
        };
        bounds.add_point(&self.center);
        bounds.add_plane(pm, p, &[]);
    }
}

impl GeoShape for GeoStandardCircle {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        match &self.circle_plane {
            None => false,
            Some(c) => c.intersects(
                &self.planet_model,
                p,
                notable_points,
                CIRCLE_POINTS,
                bounds,
                &[],
            ),
        }
    }
}

impl GeoAreaShape for GeoStandardCircle {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        match &self.circle_plane {
            None => false,
            Some(c) => geo_shape.intersects(c, CIRCLE_POINTS, &[]),
        }
    }
}

impl GeoSizeable for GeoStandardCircle {
    fn radius(&self) -> f64 {
        self.cutoff_angle
    }

    fn center(&self) -> GeoPoint {
        self.center.clone()
    }
}

impl GeoCircle for GeoStandardCircle {}
