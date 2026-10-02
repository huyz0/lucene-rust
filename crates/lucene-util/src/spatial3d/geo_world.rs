//! `GeoWorld` (`org.apache.lucene.spatial3d.geom.GeoWorld`): the whole
//! planet as a box.

use super::prelude::*;

/// The whole world.
#[derive(Debug, Clone)]
pub struct GeoWorld {
    planet_model: Arc<PlanetModel>,
    origin_point: GeoPoint,
}

impl GeoWorld {
    /// `GeoWorld(planetModel)`.
    pub fn new(planet_model: &Arc<PlanetModel>) -> GeoWorld {
        GeoWorld {
            origin_point: GeoPoint::with_magnitude(planet_model.xy_scaling, 1.0, 0.0, 0.0),
            planet_model: planet_model.clone(),
        }
    }

    /// `GeoWorld(planetModel, InputStream)`: nothing to read.
    pub fn read(planet_model: &Arc<PlanetModel>, _input: &mut Input<'_>) -> Result<GeoWorld> {
        Ok(GeoWorld::new(planet_model))
    }
}

impl SerializableObject for GeoWorld {
    fn write(&self, _out: &mut Vec<u8>) -> Result<()> {
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(26)
    }
}

impl_planet_object!(GeoWorld);

// `GeoBaseMembershipShape.computeOutsideDistance` is 0 inside, and every
// point is inside (Java's `outsideDistance`, also 0, is never reached).
impl super::shape::GeoMembershipShape for GeoWorld {}
impl super::shape::GeoOutsideDistance for GeoWorld {
    fn compute_outside_distance(&self, _style: DistanceStyle, _x: f64, _y: f64, _z: f64) -> f64 {
        0.0
    }
}

impl Membership for GeoWorld {
    fn is_within_xyz(&self, _x: f64, _y: f64, _z: f64) -> bool {
        true
    }
}

impl Bounded for GeoWorld {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        // Unbounded in all directions
        base_get_bounds(self, &self.planet_model, bounds);
    }
}

impl GeoShape for GeoWorld {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&[])
    }

    fn intersects(
        &self,
        _p: &Plane,
        _notable_points: &[GeoPoint],
        _bounds: &[&dyn Membership],
    ) -> bool {
        false
    }
}

impl GeoArea for GeoWorld {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        if !path.edge_points().is_empty() {
            // Path is always within the world
            return Ok(GeoAreaRelationship::Within);
        }
        Ok(GeoAreaRelationship::Overlaps)
    }
}

impl GeoAreaShape for GeoWorld {
    fn intersects_shape(&self, _geo_shape: &dyn GeoShape) -> bool {
        false
    }
}

impl GeoSizeable for GeoWorld {
    fn radius(&self) -> f64 {
        PI
    }

    fn center(&self) -> GeoPoint {
        // Totally arbitrary
        self.origin_point.clone()
    }
}

impl GeoBBox for GeoWorld {
    fn expand(&self, _angle: f64) -> Result<Arc<dyn GeoBBox>> {
        Ok(Arc::new(self.clone()))
    }
}
