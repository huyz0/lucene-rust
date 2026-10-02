//! `GeoS2Shape` and `GeoS2ShapeFactory`
//! (`org.apache.lucene.spatial3d.geom`): the four-cornered cell of a
//! Google S2 grid -- a convex polygon whose edges are great circles.
//! `GeoPointShapeFactory` is here too.

use super::geo_degenerate_point::GeoDegeneratePoint;
use super::prelude::*;
use super::shape::{GeoPointShape, GeoPolygon};
use super::standard_objects::{read_object_without_planet, write_object};

/// An S2 cell: four points, four sided planes.
#[derive(Debug, Clone)]
pub struct GeoS2Shape {
    planet_model: Arc<PlanetModel>,
    point1: GeoPoint,
    point2: GeoPoint,
    point3: GeoPoint,
    point4: GeoPoint,
    plane1: SidedPlane,
    plane2: SidedPlane,
    plane3: SidedPlane,
    plane4: SidedPlane,
    plane1_points: [GeoPoint; 2],
    plane2_points: [GeoPoint; 2],
    plane3_points: [GeoPoint; 2],
    plane4_points: [GeoPoint; 2],
    edge_points: [GeoPoint; 1],
}

impl GeoS2Shape {
    /// `GeoS2Shape(planetModel, point1, point2, point3, point4)`: the points
    /// in counter-clockwise order.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        point1: GeoPoint,
        point2: GeoPoint,
        point3: GeoPoint,
        point4: GeoPoint,
    ) -> Result<GeoS2Shape> {
        let plane1 = SidedPlane::from_vectors(&point4, &point1, &point2)?;
        let plane2 = SidedPlane::from_vectors(&point1, &point2, &point3)?;
        let plane3 = SidedPlane::from_vectors(&point2, &point3, &point4)?;
        let plane4 = SidedPlane::from_vectors(&point3, &point4, &point1)?;
        Ok(GeoS2Shape {
            planet_model: planet_model.clone(),
            plane1_points: [point1.clone(), point2.clone()],
            plane2_points: [point2.clone(), point3.clone()],
            plane3_points: [point3.clone(), point4.clone()],
            plane4_points: [point4.clone(), point1.clone()],
            edge_points: [point1.clone()],
            point1,
            point2,
            point3,
            point4,
            plane1,
            plane2,
            plane3,
            plane4,
        })
    }

    /// `GeoS2Shape(planetModel, InputStream)`: four points, each with its
    /// class.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<GeoS2Shape> {
        let p1 = read_object_without_planet(input)?.into_point()?;
        let p2 = read_object_without_planet(input)?.into_point()?;
        let p3 = read_object_without_planet(input)?.into_point()?;
        let p4 = read_object_without_planet(input)?.into_point()?;
        GeoS2Shape::new(planet_model, p1, p2, p3, p4)
    }

    /// `outsideDistance(distanceStyle, x, y, z)`.
    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let pm = &*self.planet_model;
        let plane_distance1 = style.compute_distance_to_plane(
            pm,
            &self.plane1,
            x,
            y,
            z,
            &[&self.plane2, &self.plane4],
        );
        let plane_distance2 = style.compute_distance_to_plane(
            pm,
            &self.plane2,
            x,
            y,
            z,
            &[&self.plane3, &self.plane1],
        );
        let plane_distance3 = style.compute_distance_to_plane(
            pm,
            &self.plane3,
            x,
            y,
            z,
            &[&self.plane4, &self.plane2],
        );
        let plane_distance4 = style.compute_distance_to_plane(
            pm,
            &self.plane4,
            x,
            y,
            z,
            &[&self.plane1, &self.plane3],
        );
        let point_distance1 = style.compute_distance(&self.point1, x, y, z);
        let point_distance2 = style.compute_distance(&self.point2, x, y, z);
        let point_distance3 = style.compute_distance(&self.point3, x, y, z);
        let point_distance4 = style.compute_distance(&self.point4, x, y, z);
        min(
            min(
                min(plane_distance1, plane_distance2),
                min(plane_distance3, plane_distance4),
            ),
            min(
                min(point_distance1, point_distance2),
                min(point_distance3, point_distance4),
            ),
        )
    }
}

impl SerializableObject for GeoS2Shape {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_object(out, &self.point1)?;
        write_object(out, &self.point2)?;
        write_object(out, &self.point3)?;
        write_object(out, &self.point4)
    }

    fn class_code(&self) -> Option<u8> {
        Some(38)
    }
}

impl_planet_object!(GeoS2Shape);
impl_membership_shape!(GeoS2Shape);
impl_base_area!(GeoS2Shape);

impl GeoPolygon for GeoS2Shape {}

impl Membership for GeoS2Shape {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.plane1.is_within_xyz(x, y, z)
            && self.plane2.is_within_xyz(x, y, z)
            && self.plane3.is_within_xyz(x, y, z)
            && self.plane4.is_within_xyz(x, y, z)
    }
}

impl Bounded for GeoS2Shape {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds
            .add_plane(pm, &self.plane1, &[&self.plane2, &self.plane4])
            .add_plane(pm, &self.plane2, &[&self.plane3, &self.plane1])
            .add_plane(pm, &self.plane3, &[&self.plane4, &self.plane2])
            .add_plane(pm, &self.plane4, &[&self.plane1, &self.plane3])
            .add_point(&self.point1)
            .add_point(&self.point2)
            .add_point(&self.point3)
            .add_point(&self.point4);
    }
}

impl GeoShape for GeoS2Shape {
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
            &self.plane1,
            notable_points,
            &self.plane1_points,
            bounds,
            &[&self.plane2, &self.plane4],
        ) || p.intersects(
            pm,
            &self.plane2,
            notable_points,
            &self.plane2_points,
            bounds,
            &[&self.plane3, &self.plane1],
        ) || p.intersects(
            pm,
            &self.plane3,
            notable_points,
            &self.plane3_points,
            bounds,
            &[&self.plane4, &self.plane2],
        ) || p.intersects(
            pm,
            &self.plane4,
            notable_points,
            &self.plane4_points,
            bounds,
            &[&self.plane1, &self.plane3],
        )
    }
}

impl GeoAreaShape for GeoS2Shape {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        geo_shape.intersects(
            &self.plane1,
            &self.plane1_points,
            &[&self.plane2, &self.plane4],
        ) || geo_shape.intersects(
            &self.plane2,
            &self.plane2_points,
            &[&self.plane3, &self.plane1],
        ) || geo_shape.intersects(
            &self.plane3,
            &self.plane3_points,
            &[&self.plane4, &self.plane2],
        ) || geo_shape.intersects(
            &self.plane4,
            &self.plane4_points,
            &[&self.plane1, &self.plane3],
        )
    }
}

/// `GeoS2ShapeFactory.makeGeoS2Shape(planetModel, point1, ..., point4)`.
pub fn make_geo_s2_shape(
    planet_model: &Arc<PlanetModel>,
    point1: GeoPoint,
    point2: GeoPoint,
    point3: GeoPoint,
    point4: GeoPoint,
) -> Result<Arc<dyn GeoPolygon>> {
    GeoS2Shape::new(planet_model, point1, point2, point3, point4).map(|s| Arc::new(s) as _)
}

/// `GeoPointShapeFactory.makeGeoPointShape(planetModel, lat, lon)`.
pub fn make_geo_point_shape(
    planet_model: &Arc<PlanetModel>,
    lat: f64,
    lon: f64,
) -> Result<Arc<dyn GeoPointShape>> {
    GeoDegeneratePoint::new(planet_model, lat, lon).map(|s| Arc::new(s) as _)
}
