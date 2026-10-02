//! `GeoDegeneratePoint` (`org.apache.lucene.spatial3d.geom.GeoDegeneratePoint`):
//! a single point as a shape -- both a zero-radius circle and a zero-size
//! box. Java's class extends `GeoPoint`; here it holds one.

use super::geo_bbox_factory::make_geo_bbox;
use super::prelude::*;
use super::shape::{
    GeoCircle, GeoDistance, GeoDistanceShape, GeoMembershipShape, GeoOutsideDistance, GeoPointShape,
};

/// A point shape.
#[derive(Debug, Clone)]
pub struct GeoDegeneratePoint {
    planet_model: Arc<PlanetModel>,
    point: [GeoPoint; 1],
}

impl GeoDegeneratePoint {
    /// `GeoDegeneratePoint(planetModel, lat, lon)`.
    pub fn new(planet_model: &Arc<PlanetModel>, lat: f64, lon: f64) -> Result<GeoDegeneratePoint> {
        Ok(GeoDegeneratePoint {
            point: [GeoPoint::from_lat_lon(planet_model, lat, lon)?],
            planet_model: planet_model.clone(),
        })
    }

    /// `GeoDegeneratePoint(planetModel, InputStream)`: a serialized
    /// `GeoPoint`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<GeoDegeneratePoint> {
        Ok(GeoDegeneratePoint {
            point: [GeoPoint::read(input)?],
            planet_model: planet_model.clone(),
        })
    }

    /// The point.
    pub fn point(&self) -> &GeoPoint {
        &self.point[0]
    }
}

impl SerializableObject for GeoDegeneratePoint {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        self.point[0].write(out);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(10)
    }
}

impl_planet_object!(GeoDegeneratePoint);

impl Membership for GeoDegeneratePoint {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.point[0].is_identical_xyz(x, y, z)
    }
}

impl Bounded for GeoDegeneratePoint {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        bounds.add_point(&self.point[0]);
    }
}

impl GeoShape for GeoDegeneratePoint {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.point)
    }

    fn intersects(
        &self,
        plane: &Plane,
        _notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        // If not on the plane, no intersection
        if !plane.evaluate_is_zero(&self.point[0]) {
            return false;
        }
        bounds.iter().all(|m| m.is_within(&self.point[0]))
    }
}

impl GeoOutsideDistance for GeoDegeneratePoint {
    fn compute_outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance(&self.point[0], x, y, z)
    }

    fn compute_outside_distance_to(&self, style: DistanceStyle, point: &GeoPoint) -> f64 {
        style.compute_distance_points(&self.point[0], point)
    }
}

impl GeoMembershipShape for GeoDegeneratePoint {}

impl GeoArea for GeoDegeneratePoint {
    fn get_relationship(&self, shape: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        if shape.is_within(&self.point[0]) {
            return Ok(GeoAreaRelationship::Contains);
        }
        Ok(GeoAreaRelationship::Disjoint)
    }
}

impl GeoAreaShape for GeoDegeneratePoint {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.is_within(&self.point[0])
    }
}

impl GeoSizeable for GeoDegeneratePoint {
    fn radius(&self) -> f64 {
        0.0
    }

    fn center(&self) -> GeoPoint {
        self.point[0].clone()
    }
}

impl GeoBBox for GeoDegeneratePoint {
    fn expand(&self, angle: f64) -> Result<Arc<dyn GeoBBox>> {
        // Java reads the `latitude`/`longitude` fields, which every
        // constructor sets.
        let p = &self.point[0];
        make_geo_bbox(
            &self.planet_model,
            p.latitude() + angle,
            p.latitude() - angle,
            p.longitude() - angle,
            p.longitude() + angle,
        )
    }
}

impl GeoDistance for GeoDegeneratePoint {
    fn compute_distance(&self, _style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        if self.is_within_xyz(x, y, z) {
            return 0.0;
        }
        f64::INFINITY
    }
}

impl GeoDistanceShape for GeoDegeneratePoint {
    fn get_distance_bounds(
        &self,
        bounds: &mut dyn Bounds,
        _style: DistanceStyle,
        _distance_value: f64,
    ) -> Result<()> {
        self.get_bounds(bounds);
        Ok(())
    }
}

impl GeoCircle for GeoDegeneratePoint {}
impl GeoPointShape for GeoDegeneratePoint {}
