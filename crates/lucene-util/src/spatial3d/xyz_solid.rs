//! The x/y/z solids (`org.apache.lucene.spatial3d.geom`'s `XYZSolid`,
//! `BaseXYZSolid`, `StandardXYZSolid`, `dXYZSolid`, `XdYZSolid`, `XYdZSolid`,
//! `dXdYZSolid`, `dXYdZSolid`, `XdYdZSolid`, `dXdYdZSolid`) and
//! `XYZSolidFactory`: an axis-aligned box in x/y/z intersected with the
//! planet's surface, the area `PointInGeo3DShapeQuery` relates each BKD
//! cell to the query shape through. A `d` in the name marks a degenerate
//! dimension (min == max), where the box face becomes a plane the surface
//! points must lie on.

#![allow(non_snake_case)]

use super::prelude::*;
use super::shape::GeoAreaObject;
use super::xyz_bounds::XYZBounds;

/// `XYZSolid`: a `GeoArea` that is also a `PlanetObject`.
pub trait XYZSolid: GeoAreaObject {}

const X_UNIT_VECTOR: Vector = Vector::new(1.0, 0.0, 0.0);
const Y_UNIT_VECTOR: Vector = Vector::new(0.0, 1.0, 0.0);
const Z_UNIT_VECTOR: Vector = Vector::new(0.0, 0.0, 1.0);
/// `BaseXYZSolid.xVerticalPlane`.
const X_VERTICAL_PLANE: Plane = Plane::new(0.0, 1.0, 0.0, 0.0);
/// `BaseXYZSolid.yVerticalPlane`.
const Y_VERTICAL_PLANE: Plane = Plane::new(1.0, 0.0, 0.0, 0.0);

/// `BaseXYZSolid`'s `ALL_INSIDE`/`SOME_INSIDE`/`NONE_INSIDE`/`NO_EDGEPOINTS`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Inside {
    All,
    Some,
    None,
    NoEdgePoints,
}

fn count_inside(points: &[GeoPoint], within: impl Fn(&GeoPoint) -> bool) -> Inside {
    if points.is_empty() {
        return Inside::NoEdgePoints;
    }
    let mut found_outside = false;
    let mut found_inside = false;
    for p in points {
        if within(p) {
            found_inside = true;
        } else {
            found_outside = true;
        }
        if found_inside && found_outside {
            return Inside::Some;
        }
    }
    if !found_inside && !found_outside {
        return Inside::None;
    }
    if found_inside && !found_outside {
        return Inside::All;
    }
    if found_outside && !found_inside {
        return Inside::None;
    }
    Inside::Some
}

/// The `getRelationship` every solid shares, with the face test `edges`
/// (none for the solids whose faces are points).
fn solid_relationship(
    solid: &dyn Membership,
    edge_points: &[GeoPoint],
    path: &dyn GeoShape,
    edges: impl FnOnce() -> bool,
) -> GeoAreaRelationship {
    // `isShapeInsideArea(path)`
    let inside_rectangle = count_inside(&path.edge_points(), |p| solid.is_within(p));
    if inside_rectangle == Inside::Some {
        return GeoAreaRelationship::Overlaps;
    }
    // `isAreaInsideShape(path)`
    let inside_shape = count_inside(edge_points, |p| path.is_within(p));
    if inside_shape == Inside::Some {
        return GeoAreaRelationship::Overlaps;
    }
    if inside_rectangle == Inside::All && inside_shape == Inside::All {
        return GeoAreaRelationship::Overlaps;
    }
    if edges() {
        return GeoAreaRelationship::Overlaps;
    }
    if inside_rectangle == Inside::All {
        return GeoAreaRelationship::Within;
    }
    if inside_shape == Inside::All {
        return GeoAreaRelationship::Contains;
    }
    GeoAreaRelationship::Disjoint
}

/// `findIntersections` with varargs bounds: `null` cannot arise for the
/// solids' perpendicular planes, and Java would dereference it.
fn intersections(
    pm: &PlanetModel,
    p: &Plane,
    q: &Plane,
    bounds: &[&dyn Membership],
) -> Vec<GeoPoint> {
    p.find_intersections(pm, q, bounds).unwrap_or_default()
}

/// The face edge point a solid adds when a face plane cuts the planet
/// without any edge (`minXEdges` and friends): a sample intersection with
/// the vertical plane.
fn face_edge(pm: &PlanetModel, face: &Plane, vertical: &Plane, applies: bool) -> Vec<GeoPoint> {
    if applies {
        face.sample_intersection_point(pm, vertical)
            .into_iter()
            .collect()
    } else {
        Vec::new()
    }
}

fn glue(arrays: &[&[GeoPoint]]) -> Vec<GeoPoint> {
    arrays.iter().flat_map(|a| a.iter().cloned()).collect()
}

macro_rules! solid_boilerplate {
    ($t:ty, $code:expr, [$($f:ident),*]) => {
        impl SerializableObject for $t {
            fn write(&self, out: &mut Vec<u8>) -> Result<()> {
                $(write_double(out, self.$f);)*
                Ok(())
            }

            fn class_code(&self) -> Option<u8> {
                Some($code)
            }
        }
        impl_planet_object_only!($t);
        impl XYZSolid for $t {}
    };
}

/// `PlanetObject` without `GeoBounds` (a solid is not a shape).
macro_rules! impl_planet_object_only {
    ($t:ty) => {
        impl $crate::spatial3d::shape::PlanetObject for $t {
            fn planet_model(&self) -> &Arc<PlanetModel> {
                &self.planet_model
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }
    };
}

/// A non-degenerate box.
#[derive(Debug, Clone)]
pub struct StandardXYZSolid {
    planet_model: Arc<PlanetModel>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
    /// `None` when the box contains the whole world (`isWholeWorld`).
    faces: Option<Box<StandardFaces>>,
}

#[derive(Debug, Clone)]
struct StandardFaces {
    min_x_plane: SidedPlane,
    max_x_plane: SidedPlane,
    min_y_plane: SidedPlane,
    max_y_plane: SidedPlane,
    min_z_plane: SidedPlane,
    max_z_plane: SidedPlane,
    min_x_plane_intersects: bool,
    max_x_plane_intersects: bool,
    min_y_plane_intersects: bool,
    max_y_plane_intersects: bool,
    min_z_plane_intersects: bool,
    max_z_plane_intersects: bool,
    edge_points: Vec<GeoPoint>,
    notable_min_x_points: Vec<GeoPoint>,
    notable_max_x_points: Vec<GeoPoint>,
    notable_min_y_points: Vec<GeoPoint>,
    notable_max_y_points: Vec<GeoPoint>,
    notable_min_z_points: Vec<GeoPoint>,
    notable_max_z_points: Vec<GeoPoint>,
}

impl StandardXYZSolid {
    /// `StandardXYZSolid(planetModel, minX, maxX, minY, maxY, minZ, maxZ)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        min_z: f64,
        max_z: f64,
    ) -> Result<StandardXYZSolid> {
        let pm = &**planet_model;
        if max_x - min_x < MINIMUM_RESOLUTION {
            return Err(illegal("X values in wrong order or identical"));
        }
        if max_y - min_y < MINIMUM_RESOLUTION {
            return Err(illegal("Y values in wrong order or identical"));
        }
        if max_z - min_z < MINIMUM_RESOLUTION {
            return Err(illegal("Z values in wrong order or identical"));
        }
        let world_min_x = pm.minimum_x_value();
        let world_max_x = pm.maximum_x_value();
        let world_min_y = pm.minimum_y_value();
        let world_max_y = pm.maximum_y_value();
        let world_min_z = pm.minimum_z_value();
        let world_max_z = pm.maximum_z_value();
        let is_whole_world = (min_x - world_min_x < -MINIMUM_RESOLUTION)
            && (max_x - world_max_x > MINIMUM_RESOLUTION)
            && (min_y - world_min_y < -MINIMUM_RESOLUTION)
            && (max_y - world_max_y > MINIMUM_RESOLUTION)
            && (min_z - world_min_z < -MINIMUM_RESOLUTION)
            && (max_z - world_max_z > MINIMUM_RESOLUTION);
        let faces = if is_whole_world {
            None
        } else {
            let min_x_plane = SidedPlane::from_xyz_normal(max_x, 0.0, 0.0, &X_UNIT_VECTOR, -min_x)?;
            let max_x_plane = SidedPlane::from_xyz_normal(min_x, 0.0, 0.0, &X_UNIT_VECTOR, -max_x)?;
            let min_y_plane = SidedPlane::from_xyz_normal(0.0, max_y, 0.0, &Y_UNIT_VECTOR, -min_y)?;
            let max_y_plane = SidedPlane::from_xyz_normal(0.0, min_y, 0.0, &Y_UNIT_VECTOR, -max_y)?;
            let min_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, max_z, &Z_UNIT_VECTOR, -min_z)?;
            let max_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, min_z, &Z_UNIT_VECTOR, -max_z)?;
            let (nx, xx, ny, xy, nz, xz) = (
                &min_x_plane,
                &max_x_plane,
                &min_y_plane,
                &max_y_plane,
                &min_z_plane,
                &max_z_plane,
            );
            let min_x_min_y = intersections(pm, nx, ny, &[xx, xy, nz, xz]);
            let min_x_max_y = intersections(pm, nx, xy, &[xx, ny, nz, xz]);
            let min_x_min_z = intersections(pm, nx, nz, &[xx, xz, ny, xy]);
            let min_x_max_z = intersections(pm, nx, xz, &[xx, nz, ny, xy]);
            let max_x_min_y = intersections(pm, xx, ny, &[nx, xy, nz, xz]);
            let max_x_max_y = intersections(pm, xx, xy, &[nx, ny, nz, xz]);
            let max_x_min_z = intersections(pm, xx, nz, &[nx, xz, ny, xy]);
            let max_x_max_z = intersections(pm, xx, xz, &[nx, nz, ny, xy]);
            let min_y_min_z = intersections(pm, ny, nz, &[xy, xz, nx, xx]);
            let min_y_max_z = intersections(pm, ny, xz, &[xy, nz, nx, xx]);
            let max_y_min_z = intersections(pm, xy, nz, &[ny, xz, nx, xx]);
            let max_y_max_z = intersections(pm, xy, xz, &[ny, nz, nx, xx]);
            let notable_min_x_points =
                glue(&[&min_x_min_y, &min_x_max_y, &min_x_min_z, &min_x_max_z]);
            let notable_max_x_points =
                glue(&[&max_x_min_y, &max_x_max_y, &max_x_min_z, &max_x_max_z]);
            let notable_min_y_points =
                glue(&[&min_x_min_y, &max_x_min_y, &min_y_min_z, &min_y_max_z]);
            let notable_max_y_points =
                glue(&[&min_x_max_y, &max_x_max_y, &max_y_min_z, &max_y_max_z]);
            let notable_min_z_points =
                glue(&[&min_x_min_z, &max_x_min_z, &min_y_min_z, &max_y_min_z]);
            let notable_max_z_points =
                glue(&[&min_x_max_z, &max_x_max_z, &min_y_max_z, &max_y_max_z]);
            let out = |x: f64, y: f64, z: f64| pm.point_outside_xyz(x, y, z);
            let min_x_min_y_min_z = out(min_x, min_y, min_z);
            let min_x_min_y_max_z = out(min_x, min_y, max_z);
            let min_x_max_y_min_z = out(min_x, max_y, min_z);
            let min_x_max_y_max_z = out(min_x, max_y, max_z);
            let max_x_min_y_min_z = out(max_x, min_y, min_z);
            let max_x_min_y_max_z = out(max_x, min_y, max_z);
            let max_x_max_y_min_z = out(max_x, max_y, min_z);
            let max_x_max_y_max_z = out(max_x, max_y, max_z);
            let r = MINIMUM_RESOLUTION;
            let min_x_edges = face_edge(
                pm,
                nx,
                &X_VERTICAL_PLANE,
                min_x - world_min_x >= -r
                    && min_x - world_max_x <= r
                    && min_y < 0.0
                    && max_y > 0.0
                    && min_z < 0.0
                    && max_z > 0.0
                    && min_x_min_y_min_z
                    && min_x_min_y_max_z
                    && min_x_max_y_min_z
                    && min_x_max_y_max_z,
            );
            let max_x_edges = face_edge(
                pm,
                xx,
                &X_VERTICAL_PLANE,
                max_x - world_min_x >= -r
                    && max_x - world_max_x <= r
                    && min_y < 0.0
                    && max_y > 0.0
                    && min_z < 0.0
                    && max_z > 0.0
                    && max_x_min_y_min_z
                    && max_x_min_y_max_z
                    && max_x_max_y_min_z
                    && max_x_max_y_max_z,
            );
            let min_y_edges = face_edge(
                pm,
                ny,
                &Y_VERTICAL_PLANE,
                min_y - world_min_y >= -r
                    && min_y - world_max_y <= r
                    && min_x < 0.0
                    && max_x > 0.0
                    && min_z < 0.0
                    && max_z > 0.0
                    && min_x_min_y_min_z
                    && min_x_min_y_max_z
                    && max_x_min_y_min_z
                    && max_x_min_y_max_z,
            );
            let max_y_edges = face_edge(
                pm,
                xy,
                &Y_VERTICAL_PLANE,
                max_y - world_min_y >= -r
                    && max_y - world_max_y <= r
                    && min_x < 0.0
                    && max_x > 0.0
                    && min_z < 0.0
                    && max_z > 0.0
                    && min_x_max_y_min_z
                    && min_x_max_y_max_z
                    && max_x_max_y_min_z
                    && max_x_max_y_max_z,
            );
            let min_z_edges = face_edge(
                pm,
                nz,
                &X_VERTICAL_PLANE,
                min_z - world_min_z >= -r
                    && min_z - world_max_z <= r
                    && min_x < 0.0
                    && max_x > 0.0
                    && min_y < 0.0
                    && max_y > 0.0
                    && min_x_min_y_min_z
                    && min_x_max_y_min_z
                    && max_x_min_y_min_z
                    && max_x_max_y_min_z,
            );
            let max_z_edges = face_edge(
                pm,
                xz,
                &X_VERTICAL_PLANE,
                max_z - world_min_z >= -r
                    && max_z - world_max_z <= r
                    && min_x < 0.0
                    && max_x > 0.0
                    && min_y < 0.0
                    && max_y > 0.0
                    && min_x_min_y_max_z
                    && min_x_max_y_max_z
                    && max_x_min_y_max_z
                    && max_x_max_y_max_z,
            );
            Some(Box::new(StandardFaces {
                min_x_plane_intersects: notable_min_x_points.len() + min_x_edges.len() > 0,
                max_x_plane_intersects: notable_max_x_points.len() + max_x_edges.len() > 0,
                min_y_plane_intersects: notable_min_y_points.len() + min_y_edges.len() > 0,
                max_y_plane_intersects: notable_max_y_points.len() + max_y_edges.len() > 0,
                min_z_plane_intersects: notable_min_z_points.len() + min_z_edges.len() > 0,
                max_z_plane_intersects: notable_max_z_points.len() + max_z_edges.len() > 0,
                edge_points: glue(&[
                    &min_x_min_y,
                    &min_x_max_y,
                    &min_x_min_z,
                    &min_x_max_z,
                    &max_x_min_y,
                    &max_x_max_y,
                    &max_x_min_z,
                    &max_x_max_z,
                    &min_y_min_z,
                    &min_y_max_z,
                    &max_y_min_z,
                    &max_y_max_z,
                    &min_x_edges,
                    &max_x_edges,
                    &min_y_edges,
                    &max_y_edges,
                    &min_z_edges,
                    &max_z_edges,
                ]),
                notable_min_x_points,
                notable_max_x_points,
                notable_min_y_points,
                notable_max_y_points,
                notable_min_z_points,
                notable_max_z_points,
                min_x_plane,
                max_x_plane,
                min_y_plane,
                max_y_plane,
                min_z_plane,
                max_z_plane,
            }))
        };
        Ok(StandardXYZSolid {
            planet_model: planet_model.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
            min_z,
            max_z,
            faces,
        })
    }

    /// `StandardXYZSolid(planetModel, InputStream)`.
    pub fn read(
        planet_model: &Arc<PlanetModel>,
        input: &mut Input<'_>,
    ) -> Result<StandardXYZSolid> {
        let v = read_doubles::<6>(input)?;
        StandardXYZSolid::new(planet_model, v[0], v[1], v[2], v[3], v[4], v[5])
    }
}

/// `n` doubles, as the solids' stream constructors read them.
pub(crate) fn read_doubles<const N: usize>(input: &mut Input<'_>) -> Result<[f64; N]> {
    let mut v = [0.0; N];
    for x in &mut v {
        *x = read_double(input)?;
    }
    Ok(v)
}

solid_boilerplate!(
    StandardXYZSolid,
    34,
    [min_x, max_x, min_y, max_y, min_z, max_z]
);

impl Membership for StandardXYZSolid {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        let Some(f) = &self.faces else {
            return true;
        };
        f.min_x_plane.is_within_xyz(x, y, z)
            && f.max_x_plane.is_within_xyz(x, y, z)
            && f.min_y_plane.is_within_xyz(x, y, z)
            && f.max_y_plane.is_within_xyz(x, y, z)
            && f.min_z_plane.is_within_xyz(x, y, z)
            && f.max_z_plane.is_within_xyz(x, y, z)
    }
}

impl GeoArea for StandardXYZSolid {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        let Some(f) = &self.faces else {
            if !path.edge_points().is_empty() {
                return Ok(GeoAreaRelationship::Within);
            }
            return Ok(GeoAreaRelationship::Overlaps);
        };
        Ok(solid_relationship(self, &f.edge_points, path, || {
            (f.min_x_plane_intersects
                && path.intersects(
                    &f.min_x_plane,
                    &f.notable_min_x_points,
                    &[
                        &f.max_x_plane,
                        &f.min_y_plane,
                        &f.max_y_plane,
                        &f.min_z_plane,
                        &f.max_z_plane,
                    ],
                ))
                || (f.max_x_plane_intersects
                    && path.intersects(
                        &f.max_x_plane,
                        &f.notable_max_x_points,
                        &[
                            &f.min_x_plane,
                            &f.min_y_plane,
                            &f.max_y_plane,
                            &f.min_z_plane,
                            &f.max_z_plane,
                        ],
                    ))
                || (f.min_y_plane_intersects
                    && path.intersects(
                        &f.min_y_plane,
                        &f.notable_min_y_points,
                        &[
                            &f.max_y_plane,
                            &f.min_x_plane,
                            &f.max_x_plane,
                            &f.min_z_plane,
                            &f.max_z_plane,
                        ],
                    ))
                || (f.max_y_plane_intersects
                    && path.intersects(
                        &f.max_y_plane,
                        &f.notable_max_y_points,
                        &[
                            &f.min_y_plane,
                            &f.min_x_plane,
                            &f.max_x_plane,
                            &f.min_z_plane,
                            &f.max_z_plane,
                        ],
                    ))
                || (f.min_z_plane_intersects
                    && path.intersects(
                        &f.min_z_plane,
                        &f.notable_min_z_points,
                        &[
                            &f.max_z_plane,
                            &f.min_x_plane,
                            &f.max_x_plane,
                            &f.min_y_plane,
                            &f.max_y_plane,
                        ],
                    ))
                || (f.max_z_plane_intersects
                    && path.intersects(
                        &f.max_z_plane,
                        &f.notable_max_z_points,
                        &[
                            &f.min_z_plane,
                            &f.min_x_plane,
                            &f.max_x_plane,
                            &f.min_y_plane,
                            &f.max_y_plane,
                        ],
                    ))
        }))
    }
}

/// A box degenerate in x.
#[derive(Debug, Clone)]
pub struct DXYZSolid {
    planet_model: Arc<PlanetModel>,
    X: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
    x_plane: Plane,
    min_y_plane: SidedPlane,
    max_y_plane: SidedPlane,
    min_z_plane: SidedPlane,
    max_z_plane: SidedPlane,
    edge_points: Vec<GeoPoint>,
    notable_x_points: Vec<GeoPoint>,
}

impl DXYZSolid {
    /// `dXYZSolid(planetModel, X, minY, maxY, minZ, maxZ)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        X: f64,
        min_y: f64,
        max_y: f64,
        min_z: f64,
        max_z: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if max_y - min_y < MINIMUM_RESOLUTION {
            return Err(illegal("Y values in wrong order or identical"));
        }
        if max_z - min_z < MINIMUM_RESOLUTION {
            return Err(illegal("Z values in wrong order or identical"));
        }
        let world_min_x = pm.minimum_x_value();
        let world_max_x = pm.maximum_x_value();
        let x_plane = Plane::with_d(&X_UNIT_VECTOR, -X);
        let min_y_plane = SidedPlane::from_xyz_normal(0.0, max_y, 0.0, &Y_UNIT_VECTOR, -min_y)?;
        let max_y_plane = SidedPlane::from_xyz_normal(0.0, min_y, 0.0, &Y_UNIT_VECTOR, -max_y)?;
        let min_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, max_z, &Z_UNIT_VECTOR, -min_z)?;
        let max_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, min_z, &Z_UNIT_VECTOR, -max_z)?;
        let x_min_y = intersections(
            pm,
            &x_plane,
            &min_y_plane,
            &[&max_y_plane, &min_z_plane, &max_z_plane],
        );
        let x_max_y = intersections(
            pm,
            &x_plane,
            &max_y_plane,
            &[&min_y_plane, &min_z_plane, &max_z_plane],
        );
        let x_min_z = intersections(
            pm,
            &x_plane,
            &min_z_plane,
            &[&max_z_plane, &min_y_plane, &max_y_plane],
        );
        let x_max_z = intersections(
            pm,
            &x_plane,
            &max_z_plane,
            &[&min_z_plane, &min_y_plane, &max_y_plane],
        );
        let notable_x_points = glue(&[&x_min_y, &x_max_y, &x_min_z, &x_max_z]);
        let out = |x: f64, y: f64, z: f64| pm.point_outside_xyz(x, y, z);
        let x_edges = face_edge(
            pm,
            &x_plane,
            &X_VERTICAL_PLANE,
            X - world_min_x >= -MINIMUM_RESOLUTION
                && X - world_max_x <= MINIMUM_RESOLUTION
                && min_y < 0.0
                && max_y > 0.0
                && min_z < 0.0
                && max_z > 0.0
                && out(X, min_y, min_z)
                && out(X, min_y, max_z)
                && out(X, max_y, min_z)
                && out(X, max_y, max_z),
        );
        Ok(DXYZSolid {
            planet_model: planet_model.clone(),
            X,
            min_y,
            max_y,
            min_z,
            max_z,
            edge_points: glue(&[&x_min_y, &x_max_y, &x_min_z, &x_max_z, &x_edges]),
            notable_x_points,
            x_plane,
            min_y_plane,
            max_y_plane,
            min_z_plane,
            max_z_plane,
        })
    }

    /// `dXYZSolid(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let v = read_doubles::<5>(input)?;
        Self::new(planet_model, v[0], v[1], v[2], v[3], v[4])
    }
}

solid_boilerplate!(DXYZSolid, 30, [X, min_y, max_y, min_z, max_z]);

impl Membership for DXYZSolid {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.x_plane.evaluate_is_zero_xyz(x, y, z)
            && self.min_y_plane.is_within_xyz(x, y, z)
            && self.max_y_plane.is_within_xyz(x, y, z)
            && self.min_z_plane.is_within_xyz(x, y, z)
            && self.max_z_plane.is_within_xyz(x, y, z)
    }
}

impl GeoArea for DXYZSolid {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(solid_relationship(self, &self.edge_points, path, || {
            path.intersects(
                &self.x_plane,
                &self.notable_x_points,
                &[
                    &self.min_y_plane,
                    &self.max_y_plane,
                    &self.min_z_plane,
                    &self.max_z_plane,
                ],
            )
        }))
    }
}

/// A box degenerate in y.
#[derive(Debug, Clone)]
pub struct XdYZSolid {
    planet_model: Arc<PlanetModel>,
    min_x: f64,
    max_x: f64,
    Y: f64,
    min_z: f64,
    max_z: f64,
    min_x_plane: SidedPlane,
    max_x_plane: SidedPlane,
    y_plane: Plane,
    min_z_plane: SidedPlane,
    max_z_plane: SidedPlane,
    edge_points: Vec<GeoPoint>,
    notable_y_points: Vec<GeoPoint>,
}

impl XdYZSolid {
    /// `XdYZSolid(planetModel, minX, maxX, Y, minZ, maxZ)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        min_x: f64,
        max_x: f64,
        Y: f64,
        min_z: f64,
        max_z: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if max_x - min_x < MINIMUM_RESOLUTION {
            return Err(illegal("X values in wrong order or identical"));
        }
        if max_z - min_z < MINIMUM_RESOLUTION {
            return Err(illegal("Z values in wrong order or identical"));
        }
        let world_min_y = pm.minimum_y_value();
        let world_max_y = pm.maximum_y_value();
        let min_x_plane = SidedPlane::from_xyz_normal(max_x, 0.0, 0.0, &X_UNIT_VECTOR, -min_x)?;
        let max_x_plane = SidedPlane::from_xyz_normal(min_x, 0.0, 0.0, &X_UNIT_VECTOR, -max_x)?;
        let y_plane = Plane::with_d(&Y_UNIT_VECTOR, -Y);
        let min_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, max_z, &Z_UNIT_VECTOR, -min_z)?;
        let max_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, min_z, &Z_UNIT_VECTOR, -max_z)?;
        let min_x_y = intersections(
            pm,
            &min_x_plane,
            &y_plane,
            &[&max_x_plane, &min_z_plane, &max_z_plane],
        );
        let max_x_y = intersections(
            pm,
            &max_x_plane,
            &y_plane,
            &[&min_x_plane, &min_z_plane, &max_z_plane],
        );
        let y_min_z = intersections(
            pm,
            &y_plane,
            &min_z_plane,
            &[&max_z_plane, &min_x_plane, &max_x_plane],
        );
        let y_max_z = intersections(
            pm,
            &y_plane,
            &max_z_plane,
            &[&min_z_plane, &min_x_plane, &max_x_plane],
        );
        let notable_y_points = glue(&[&min_x_y, &max_x_y, &y_min_z, &y_max_z]);
        let out = |x: f64, y: f64, z: f64| pm.point_outside_xyz(x, y, z);
        let y_edges = face_edge(
            pm,
            &y_plane,
            &Y_VERTICAL_PLANE,
            Y - world_min_y >= -MINIMUM_RESOLUTION
                && Y - world_max_y <= MINIMUM_RESOLUTION
                && min_x < 0.0
                && max_x > 0.0
                && min_z < 0.0
                && max_z > 0.0
                && out(min_x, Y, min_z)
                && out(min_x, Y, max_z)
                && out(max_x, Y, min_z)
                && out(max_x, Y, max_z),
        );
        Ok(XdYZSolid {
            planet_model: planet_model.clone(),
            min_x,
            max_x,
            Y,
            min_z,
            max_z,
            edge_points: glue(&[&min_x_y, &max_x_y, &y_min_z, &y_max_z, &y_edges]),
            notable_y_points,
            min_x_plane,
            max_x_plane,
            y_plane,
            min_z_plane,
            max_z_plane,
        })
    }

    /// `XdYZSolid(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let v = read_doubles::<5>(input)?;
        Self::new(planet_model, v[0], v[1], v[2], v[3], v[4])
    }
}

solid_boilerplate!(XdYZSolid, 32, [min_x, max_x, Y, min_z, max_z]);

impl Membership for XdYZSolid {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.min_x_plane.is_within_xyz(x, y, z)
            && self.max_x_plane.is_within_xyz(x, y, z)
            && self.y_plane.evaluate_is_zero_xyz(x, y, z)
            && self.min_z_plane.is_within_xyz(x, y, z)
            && self.max_z_plane.is_within_xyz(x, y, z)
    }
}

impl GeoArea for XdYZSolid {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(solid_relationship(self, &self.edge_points, path, || {
            path.intersects(
                &self.y_plane,
                &self.notable_y_points,
                &[
                    &self.min_x_plane,
                    &self.max_x_plane,
                    &self.min_z_plane,
                    &self.max_z_plane,
                ],
            )
        }))
    }
}

/// A box degenerate in z.
#[derive(Debug, Clone)]
pub struct XYdZSolid {
    planet_model: Arc<PlanetModel>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    Z: f64,
    min_x_plane: SidedPlane,
    max_x_plane: SidedPlane,
    min_y_plane: SidedPlane,
    max_y_plane: SidedPlane,
    z_plane: Plane,
    edge_points: Vec<GeoPoint>,
    notable_z_points: Vec<GeoPoint>,
}

impl XYdZSolid {
    /// `XYdZSolid(planetModel, minX, maxX, minY, maxY, Z)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
        Z: f64,
    ) -> Result<Self> {
        let pm = &**planet_model;
        if max_x - min_x < MINIMUM_RESOLUTION {
            return Err(illegal("X values in wrong order or identical"));
        }
        if max_y - min_y < MINIMUM_RESOLUTION {
            return Err(illegal("Y values in wrong order or identical"));
        }
        let world_min_z = pm.minimum_z_value();
        let world_max_z = pm.maximum_z_value();
        let min_x_plane = SidedPlane::from_xyz_normal(max_x, 0.0, 0.0, &X_UNIT_VECTOR, -min_x)?;
        let max_x_plane = SidedPlane::from_xyz_normal(min_x, 0.0, 0.0, &X_UNIT_VECTOR, -max_x)?;
        let min_y_plane = SidedPlane::from_xyz_normal(0.0, max_y, 0.0, &Y_UNIT_VECTOR, -min_y)?;
        let max_y_plane = SidedPlane::from_xyz_normal(0.0, min_y, 0.0, &Y_UNIT_VECTOR, -max_y)?;
        let z_plane = Plane::with_d(&Z_UNIT_VECTOR, -Z);
        let min_x_z = intersections(
            pm,
            &min_x_plane,
            &z_plane,
            &[&max_x_plane, &min_y_plane, &max_y_plane],
        );
        let max_x_z = intersections(
            pm,
            &max_x_plane,
            &z_plane,
            &[&min_x_plane, &min_y_plane, &max_y_plane],
        );
        let min_y_z = intersections(
            pm,
            &min_y_plane,
            &z_plane,
            &[&max_y_plane, &min_x_plane, &max_x_plane],
        );
        let max_y_z = intersections(
            pm,
            &max_y_plane,
            &z_plane,
            &[&min_y_plane, &min_x_plane, &max_x_plane],
        );
        let notable_z_points = glue(&[&min_x_z, &max_x_z, &min_y_z, &max_y_z]);
        let out = |x: f64, y: f64, z: f64| pm.point_outside_xyz(x, y, z);
        let z_edges = face_edge(
            pm,
            &z_plane,
            &X_VERTICAL_PLANE,
            Z - world_min_z >= -MINIMUM_RESOLUTION
                && Z - world_max_z <= MINIMUM_RESOLUTION
                && min_x < 0.0
                && max_x > 0.0
                && min_y < 0.0
                && max_y > 0.0
                && out(min_x, min_y, Z)
                && out(min_x, max_y, Z)
                && out(max_x, min_y, Z)
                && out(max_x, max_y, Z),
        );
        Ok(XYdZSolid {
            planet_model: planet_model.clone(),
            min_x,
            max_x,
            min_y,
            max_y,
            Z,
            edge_points: glue(&[&min_x_z, &max_x_z, &min_y_z, &max_y_z, &z_edges]),
            notable_z_points,
            min_x_plane,
            max_x_plane,
            min_y_plane,
            max_y_plane,
            z_plane,
        })
    }

    /// `XYdZSolid(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let v = read_doubles::<5>(input)?;
        Self::new(planet_model, v[0], v[1], v[2], v[3], v[4])
    }
}

solid_boilerplate!(XYdZSolid, 33, [min_x, max_x, min_y, max_y, Z]);

impl Membership for XYdZSolid {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.min_x_plane.is_within_xyz(x, y, z)
            && self.max_x_plane.is_within_xyz(x, y, z)
            && self.min_y_plane.is_within_xyz(x, y, z)
            && self.max_y_plane.is_within_xyz(x, y, z)
            && self.z_plane.evaluate_is_zero_xyz(x, y, z)
    }
}

impl GeoArea for XYdZSolid {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        Ok(solid_relationship(self, &self.edge_points, path, || {
            path.intersects(
                &self.z_plane,
                &self.notable_z_points,
                &[
                    &self.min_x_plane,
                    &self.max_x_plane,
                    &self.min_y_plane,
                    &self.max_y_plane,
                ],
            )
        }))
    }
}

/// The solids degenerate in two dimensions: the surface points on a line.
macro_rules! two_degenerate_solid {
    ($t:ident, $java:literal, $code:expr, $a:ident, $b:ident, $c:ident, $d:ident, $check:literal, $build:expr) => {
        #[doc = concat!("`", $java, "`: degenerate in two dimensions -- the surface points of a line segment.")]
        #[derive(Debug, Clone)]
        pub struct $t {
            planet_model: Arc<PlanetModel>,
            $a: f64,
            $b: f64,
            $c: f64,
            $d: f64,
            surface_points: Vec<GeoPoint>,
        }

        impl $t {
            #[doc = concat!("`", $java, "(planetModel, ...)`.")]
            pub fn new(planet_model: &Arc<PlanetModel>, $a: f64, $b: f64, $c: f64, $d: f64) -> Result<Self> {
                let build: fn(&PlanetModel, f64, f64, f64, f64) -> Result<Option<Vec<GeoPoint>>> = $build;
                let Some(surface_points) = build(planet_model, $a, $b, $c, $d)? else {
                    return Err(illegal($check));
                };
                Ok($t {
                    planet_model: planet_model.clone(),
                    $a,
                    $b,
                    $c,
                    $d,
                    surface_points,
                })
            }

            #[doc = concat!("`", $java, "(planetModel, InputStream)`.")]
            pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
                let v = read_doubles::<4>(input)?;
                Self::new(planet_model, v[0], v[1], v[2], v[3])
            }
        }

        solid_boilerplate!($t, $code, [$a, $b, $c, $d]);

        impl Membership for $t {
            fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
                self.surface_points.iter().any(|p| p.is_identical_xyz(x, y, z))
            }
        }

        impl GeoArea for $t {
            fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
                Ok(solid_relationship(self, &self.surface_points, path, || false))
            }
        }
    };
}

two_degenerate_solid!(
    DXdYZSolid,
    "dXdYZSolid",
    28,
    X,
    Y,
    min_z,
    max_z,
    "Z values in wrong order or identical",
    |pm, X, Y, min_z, max_z| {
        if max_z - min_z < MINIMUM_RESOLUTION {
            return Ok(None);
        }
        let x_plane = Plane::with_d(&X_UNIT_VECTOR, -X);
        let y_plane = Plane::with_d(&Y_UNIT_VECTOR, -Y);
        let min_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, max_z, &Z_UNIT_VECTOR, -min_z)?;
        let max_z_plane = SidedPlane::from_xyz_normal(0.0, 0.0, min_z, &Z_UNIT_VECTOR, -max_z)?;
        Ok(Some(intersections(
            pm,
            &x_plane,
            &y_plane,
            &[&min_z_plane, &max_z_plane],
        )))
    }
);

two_degenerate_solid!(
    DXYdZSolid,
    "dXYdZSolid",
    29,
    X,
    min_y,
    max_y,
    Z,
    "Y values in wrong order or identical",
    |pm, X, min_y, max_y, Z| {
        if max_y - min_y < MINIMUM_RESOLUTION {
            return Ok(None);
        }
        let x_plane = Plane::with_d(&X_UNIT_VECTOR, -X);
        let z_plane = Plane::with_d(&Z_UNIT_VECTOR, -Z);
        let min_y_plane = SidedPlane::from_xyz_normal(0.0, max_y, 0.0, &Y_UNIT_VECTOR, -min_y)?;
        let max_y_plane = SidedPlane::from_xyz_normal(0.0, min_y, 0.0, &Y_UNIT_VECTOR, -max_y)?;
        Ok(Some(intersections(
            pm,
            &x_plane,
            &z_plane,
            &[&min_y_plane, &max_y_plane],
        )))
    }
);

two_degenerate_solid!(
    XdYdZSolid,
    "XdYdZSolid",
    31,
    min_x,
    max_x,
    Y,
    Z,
    "X values in wrong order or identical",
    |pm, min_x, max_x, Y, Z| {
        if max_x - min_x < MINIMUM_RESOLUTION {
            return Ok(None);
        }
        let y_plane = Plane::with_d(&Y_UNIT_VECTOR, -Y);
        let z_plane = Plane::with_d(&Z_UNIT_VECTOR, -Z);
        let min_x_plane = SidedPlane::from_xyz_normal(max_x, 0.0, 0.0, &X_UNIT_VECTOR, -min_x)?;
        let max_x_plane = SidedPlane::from_xyz_normal(min_x, 0.0, 0.0, &X_UNIT_VECTOR, -max_x)?;
        Ok(Some(intersections(
            pm,
            &y_plane,
            &z_plane,
            &[&min_x_plane, &max_x_plane],
        )))
    }
);

/// `dXdYdZSolid`: degenerate in all three dimensions -- a point, which may
/// not be on the surface.
#[derive(Debug, Clone)]
pub struct DXdYdZSolid {
    planet_model: Arc<PlanetModel>,
    X: f64,
    Y: f64,
    Z: f64,
    /// The point, when it is on the surface (`isOnSurface`).
    edge_points: Vec<GeoPoint>,
}

impl DXdYdZSolid {
    /// `dXdYdZSolid(planetModel, X, Y, Z)`.
    pub fn new(planet_model: &Arc<PlanetModel>, X: f64, Y: f64, Z: f64) -> DXdYdZSolid {
        let is_on_surface = planet_model.point_on_surface_xyz(X, Y, Z);
        DXdYdZSolid {
            planet_model: planet_model.clone(),
            X,
            Y,
            Z,
            edge_points: if is_on_surface {
                vec![GeoPoint::new(X, Y, Z)]
            } else {
                Vec::new()
            },
        }
    }

    /// `dXdYdZSolid(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<Self> {
        let v = read_doubles::<3>(input)?;
        Ok(Self::new(planet_model, v[0], v[1], v[2]))
    }
}

solid_boilerplate!(DXdYdZSolid, 27, [X, Y, Z]);

impl Membership for DXdYdZSolid {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.edge_points
            .first()
            .is_some_and(|p| p.is_identical_xyz(x, y, z))
    }
}

impl GeoArea for DXdYdZSolid {
    fn get_relationship(&self, path: &dyn GeoShape) -> Result<GeoAreaRelationship> {
        if self.edge_points.is_empty() {
            return Ok(GeoAreaRelationship::Disjoint);
        }
        Ok(solid_relationship(self, &self.edge_points, path, || false))
    }
}

/// `XYZSolidFactory.makeXYZSolid(planetModel, minX, maxX, minY, maxY, minZ,
/// maxZ)`: the solid for the box, degenerate where a dimension's extent is
/// below the resolution.
pub fn make_xyz_solid(
    planet_model: &Arc<PlanetModel>,
    min_x: f64,
    max_x: f64,
    min_y: f64,
    max_y: f64,
    min_z: f64,
    max_z: f64,
) -> Result<Arc<dyn XYZSolid>> {
    let pm = planet_model;
    let r = MINIMUM_RESOLUTION;
    if abs(max_x - min_x) < r {
        if abs(max_y - min_y) < r {
            if abs(max_z - min_z) < r {
                return Ok(Arc::new(DXdYdZSolid::new(
                    pm,
                    (min_x + max_x) * 0.5,
                    (min_y + max_y) * 0.5,
                    min_z,
                )));
            }
            return DXdYZSolid::new(
                pm,
                (min_x + max_x) * 0.5,
                (min_y + max_y) * 0.5,
                min_z,
                max_z,
            )
            .map(|s| Arc::new(s) as _);
        }
        if abs(max_z - min_z) < r {
            return DXYdZSolid::new(
                pm,
                (min_x + max_x) * 0.5,
                min_y,
                max_y,
                (min_z + max_z) * 0.5,
            )
            .map(|s| Arc::new(s) as _);
        }
        return DXYZSolid::new(pm, (min_x + max_x) * 0.5, min_y, max_y, min_z, max_z)
            .map(|s| Arc::new(s) as _);
    }
    if abs(max_y - min_y) < r {
        if abs(max_z - min_z) < r {
            return XdYdZSolid::new(
                pm,
                min_x,
                max_x,
                (min_y + max_y) * 0.5,
                (min_z + max_z) * 0.5,
            )
            .map(|s| Arc::new(s) as _);
        }
        return XdYZSolid::new(pm, min_x, max_x, (min_y + max_y) * 0.5, min_z, max_z)
            .map(|s| Arc::new(s) as _);
    }
    if abs(max_z - min_z) < r {
        return XYdZSolid::new(pm, min_x, max_x, min_y, max_y, (min_z + max_z) * 0.5)
            .map(|s| Arc::new(s) as _);
    }
    StandardXYZSolid::new(pm, min_x, max_x, min_y, max_y, min_z, max_z).map(|s| Arc::new(s) as _)
}

/// `XYZSolidFactory.makeXYZSolid(planetModel, XYZBounds)`. Java unboxes a
/// missing bound (`NullPointerException`); that is an error here.
pub fn make_xyz_solid_from_bounds(
    planet_model: &Arc<PlanetModel>,
    b: &XYZBounds,
) -> Result<Arc<dyn XYZSolid>> {
    let npe = || Error::NullPointer("bounds are unset".into());
    make_xyz_solid(
        planet_model,
        b.minimum_x().ok_or_else(npe)?,
        b.maximum_x().ok_or_else(npe)?,
        b.minimum_y().ok_or_else(npe)?,
        b.maximum_y().ok_or_else(npe)?,
        b.minimum_z().ok_or_else(npe)?,
        b.maximum_z().ok_or_else(npe)?,
    )
}
