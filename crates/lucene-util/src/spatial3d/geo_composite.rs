//! The composite shapes (`org.apache.lucene.spatial3d.geom`'s
//! `GeoBaseCompositeShape`, `GeoBaseCompositeMembershipShape`,
//! `GeoBaseCompositeAreaShape`, `GeoCompositeMembershipShape`,
//! `GeoCompositeAreaShape` and `GeoCompositePolygon`): the union of a list
//! of shapes of one interface.
//!
//! Java's generic base classes become one macro instantiated per member
//! interface; what each level adds is the same code.

use super::prelude::*;
use super::shape::{
    base_get_relationship, GeoMembershipShape, GeoOutsideDistance, GeoPolygon, PlanetObject,
};
use super::standard_objects::{
    read_heterogeneous_array, write_heterogeneous_array, StandardObject,
};

/// `addShape`'s check.
fn check_member_planet(planet_model: &PlanetModel, member: &dyn PlanetObject) -> Result<()> {
    if **member.planet_model() != *planet_model {
        return Err(illegal(
            "Cannot add a shape into a composite with different planet models.",
        ));
    }
    Ok(())
}

/// `GeoBaseCompositeShape`'s and `GeoBaseCompositeMembershipShape`'s body
/// for one composite type over members `Arc<$m>`.
macro_rules! composite_shape {
    ($t:ident, $m:ty, $java:literal, $code:literal, $cast:ident) => {
        #[doc = concat!("`", $java, "`: the union of its member shapes.")]
        #[derive(Clone)]
        pub struct $t {
            planet_model: Arc<PlanetModel>,
            shapes: Vec<Arc<$m>>,
        }

        impl $t {
            #[doc = concat!("`", $java, "(planetModel)`: an empty composite.")]
            pub fn new(planet_model: &Arc<PlanetModel>) -> $t {
                $t {
                    planet_model: planet_model.clone(),
                    shapes: Vec::new(),
                }
            }

            #[doc = concat!("`", $java, "(planetModel, InputStream)`.")]
            pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<$t> {
                let mut rval = $t::new(planet_model);
                for member in read_heterogeneous_array(planet_model, input)? {
                    rval.add_shape(StandardObject::$cast(member)?)?;
                }
                Ok(rval)
            }

            /// `addShape(shape)`: fails for a shape on another planet.
            pub fn add_shape(&mut self, shape: Arc<$m>) -> Result<()> {
                check_member_planet(&self.planet_model, &*shape)?;
                self.shapes.push(shape);
                Ok(())
            }

            /// `size()`.
            pub fn size(&self) -> usize {
                self.shapes.len()
            }

            /// `getShape(index)`.
            pub fn get_shape(&self, index: usize) -> &Arc<$m> {
                &self.shapes[index]
            }

            /// `getShapes()`.
            pub fn shapes(&self) -> &[Arc<$m>] {
                &self.shapes
            }
        }

        impl SerializableObject for $t {
            fn write(&self, out: &mut Vec<u8>) -> Result<()> {
                write_heterogeneous_array(out, &self.shapes)
            }

            fn class_code(&self) -> Option<u8> {
                Some($code)
            }
        }

        impl PlanetObject for $t {
            fn planet_model(&self) -> &Arc<PlanetModel> {
                &self.planet_model
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        impl super::shape::GeoBounds for $t {}

        impl Membership for $t {
            fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
                self.shapes.iter().any(|s| s.is_within_xyz(x, y, z))
            }
        }

        impl Bounded for $t {
            fn get_bounds(&self, bounds: &mut dyn Bounds) {
                for shape in &self.shapes {
                    shape.get_bounds(bounds);
                }
            }
        }

        impl GeoShape for $t {
            fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
                let mut edge_points = Vec::new();
                for shape in &self.shapes {
                    edge_points.extend_from_slice(&shape.edge_points());
                }
                Cow::Owned(edge_points)
            }

            fn intersects(
                &self,
                p: &Plane,
                notable_points: &[GeoPoint],
                bounds: &[&dyn Membership],
            ) -> bool {
                self.shapes
                    .iter()
                    .any(|s| s.intersects(p, notable_points, bounds))
            }
        }

        impl GeoOutsideDistance for $t {
            fn compute_outside_distance(
                &self,
                style: DistanceStyle,
                x: f64,
                y: f64,
                z: f64,
            ) -> f64 {
                if self.is_within_xyz(x, y, z) {
                    return 0.0;
                }
                let mut distance = f64::INFINITY;
                for shape in &self.shapes {
                    let normal_distance = shape.compute_outside_distance(style, x, y, z);
                    if normal_distance < distance {
                        distance = normal_distance;
                    }
                }
                distance
            }
        }

        impl GeoMembershipShape for $t {}
    };
}

/// `GeoBaseCompositeAreaShape`'s additions.
macro_rules! composite_area_shape {
    ($t:ident) => {
        impl GeoArea for $t {
            fn get_relationship(&self, shape: &dyn GeoShape) -> Result<GeoAreaRelationship> {
                base_get_relationship(self, shape)
            }
        }

        impl GeoAreaShape for $t {
            fn intersects_shape(&self, shape: &dyn GeoShape) -> bool {
                self.shapes.iter().any(|s| s.intersects_shape(shape))
            }
        }
    };
}

composite_shape!(
    GeoCompositeMembershipShape,
    dyn GeoMembershipShape,
    "GeoCompositeMembershipShape",
    8,
    into_membership_shape
);
composite_shape!(
    GeoCompositeAreaShape,
    dyn GeoAreaShape,
    "GeoCompositeAreaShape",
    9,
    into_area_shape
);
composite_area_shape!(GeoCompositeAreaShape);
composite_shape!(
    GeoCompositePolygon,
    dyn GeoPolygon,
    "GeoCompositePolygon",
    7,
    into_polygon
);
composite_area_shape!(GeoCompositePolygon);
impl GeoPolygon for GeoCompositePolygon {}
