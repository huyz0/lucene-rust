//! `GeoPolygonFactory` (`org.apache.lucene.spatial3d.geom.GeoPolygonFactory`):
//! turns a ring of points (with holes) into a polygon -- a composite of
//! convex polygons and at most one concave polygon when the ring can be
//! tiled, a [`GeoComplexPolygon`] otherwise.
//!
//! The tiling compares points and edges by Java object identity (`==`).
//! Points are therefore indices into the caller's list here -- every list
//! element is its own object, as in Java when the caller passes distinct
//! `GeoPoint`s (one object passed twice in a list would be one identity in
//! Java and two here) -- and edges are indices into the edge buffer's arena.
//! `TileException` is [`Fail::Tile`].

use super::errors::catch;
use super::geo_complex_polygon::GeoComplexPolygon;
use super::geo_composite::GeoCompositePolygon;
use super::geo_convex_polygon::{points_string, GeoConcavePolygon, GeoConvexPolygon};
use super::jmath::atan2;
use super::prelude::*;
use super::shape::GeoPolygon;
use crate::java_random::JavaRandom;

/// `SMALL_POLYGON_CUTOFF_EDGES`: larger rings go straight to a complex
/// polygon.
const SMALL_POLYGON_CUTOFF_EDGES: usize = 100;

/// `PolygonDescription`: a ring and its holes.
#[derive(Debug, Clone, Default)]
pub struct PolygonDescription {
    /// The ring's points, in order (implicitly closed).
    pub points: Vec<GeoPoint>,
    /// The holes, each a description in its own right.
    pub holes: Vec<PolygonDescription>,
}

impl PolygonDescription {
    /// `PolygonDescription(points)`.
    pub fn new(points: Vec<GeoPoint>) -> PolygonDescription {
        PolygonDescription {
            points,
            holes: Vec::new(),
        }
    }

    /// `PolygonDescription(points, holes)`.
    pub fn with_holes(points: Vec<GeoPoint>, holes: Vec<PolygonDescription>) -> PolygonDescription {
        PolygonDescription { points, holes }
    }
}

/// Why tiling stopped: Java's checked `TileException`, or an exception
/// that propagates to the caller.
#[derive(Debug)]
enum Fail {
    Tile(String),
    Err(Error),
}

impl From<Error> for Fail {
    fn from(e: Error) -> Fail {
        Fail::Err(e)
    }
}

type TileResult<T> = std::result::Result<T, Fail>;

/// `makeGeoConcavePolygon(planetModel, pointList)`.
pub fn make_geo_concave_polygon(
    planet_model: &Arc<PlanetModel>,
    point_list: Vec<GeoPoint>,
) -> Result<Arc<dyn GeoPolygon>> {
    make_geo_concave_polygon_with_holes(planet_model, point_list, None)
}

/// `makeGeoConvexPolygon(planetModel, pointList)`.
pub fn make_geo_convex_polygon(
    planet_model: &Arc<PlanetModel>,
    point_list: Vec<GeoPoint>,
) -> Result<Arc<dyn GeoPolygon>> {
    make_geo_convex_polygon_with_holes(planet_model, point_list, None)
}

/// `makeGeoConcavePolygon(planetModel, pointList, holes)`.
pub fn make_geo_concave_polygon_with_holes(
    planet_model: &Arc<PlanetModel>,
    point_list: Vec<GeoPoint>,
    holes: Option<Vec<Arc<dyn GeoPolygon>>>,
) -> Result<Arc<dyn GeoPolygon>> {
    GeoConcavePolygon::with_holes(planet_model, point_list, holes).map(|s| Arc::new(s) as _)
}

/// `makeGeoConvexPolygon(planetModel, pointList, holes)`.
pub fn make_geo_convex_polygon_with_holes(
    planet_model: &Arc<PlanetModel>,
    point_list: Vec<GeoPoint>,
    holes: Option<Vec<Arc<dyn GeoPolygon>>>,
) -> Result<Arc<dyn GeoPolygon>> {
    GeoConvexPolygon::with_holes(planet_model, point_list, holes).map(|s| Arc::new(s) as _)
}

/// `makeGeoPolygon(planetModel, description)`.
pub fn make_geo_polygon_from_description(
    planet_model: &Arc<PlanetModel>,
    description: &PolygonDescription,
) -> Result<Option<Arc<dyn GeoPolygon>>> {
    make_geo_polygon_from_description_lenient(planet_model, description, 0.0)
}

/// `makeGeoPolygon(planetModel, description, leniencyValue)`: `None` for a
/// degenerate ring (Java's `null`).
pub fn make_geo_polygon_from_description_lenient(
    planet_model: &Arc<PlanetModel>,
    description: &PolygonDescription,
    leniency_value: f64,
) -> Result<Option<Arc<dyn GeoPolygon>>> {
    // First, convert the holes to polygons in their own right.
    let holes = if !description.holes.is_empty() {
        let mut holes = Vec::with_capacity(description.holes.len());
        for hole_description in &description.holes {
            let hole = make_geo_polygon_from_description_lenient(
                planet_model,
                hole_description,
                leniency_value,
            );
            match hole? {
                Some(gp) => holes.push(gp),
                None => return Ok(None),
            }
        }
        Some(holes)
    } else {
        None
    };
    if description.points.len() <= SMALL_POLYGON_CUTOFF_EDGES {
        let pts = &description.points;
        let Some(first_filtered_point_list) = filter_points(pts)? else {
            return Ok(None);
        };
        let Some(filtered_point_list) =
            filter_edges(pts, &first_filtered_point_list, leniency_value)?
        else {
            return Ok(None);
        };
        match catch_tile(planet_model, pts, &filtered_point_list, &holes) {
            Ok(polygon) => return Ok(Some(polygon)),
            Err(Fail::Err(e)) => return Err(e),
            // Couldn't tile the polygon; use GeoComplexPolygon instead, if
            // we can.
            Err(Fail::Tile(_)) => {}
        }
    }
    // Fallback: create large geo polygon, using complex polygon logic.
    make_large_geo_polygon(planet_model, std::slice::from_ref(description)).map(Some)
}

/// `makeGeoPolygon(planetModel, pointList)`.
pub fn make_geo_polygon(
    planet_model: &Arc<PlanetModel>,
    point_list: &[GeoPoint],
) -> Result<Option<Arc<dyn GeoPolygon>>> {
    make_geo_polygon_with_holes(planet_model, point_list, None, 0.0)
}

/// `makeGeoPolygon(planetModel, pointList, holes, leniencyValue)`: `None`
/// for a degenerate ring (Java's `null`).
pub fn make_geo_polygon_with_holes(
    planet_model: &Arc<PlanetModel>,
    point_list: &[GeoPoint],
    holes: Option<Vec<Arc<dyn GeoPolygon>>>,
    leniency_value: f64,
) -> Result<Option<Arc<dyn GeoPolygon>>> {
    let Some(first_filtered_point_list) = filter_points(point_list)? else {
        return Ok(None);
    };
    let Some(filtered_point_list) =
        filter_edges(point_list, &first_filtered_point_list, leniency_value)?
    else {
        return Ok(None);
    };
    match catch_tile(planet_model, point_list, &filtered_point_list, &holes) {
        Ok(polygon) => Ok(Some(polygon)),
        Err(Fail::Err(e)) => Err(e),
        Err(Fail::Tile(msg)) => {
            // Couldn't tile the polygon; use GeoComplexPolygon instead, if
            // we can.
            if holes.as_ref().is_some_and(|h| !h.is_empty()) {
                // We currently cannot get the list of points that went into
                // making a hole back out, so don't allow this case.
                return Err(Error::IllegalArgument(msg));
            }
            let description = PolygonDescription::new(point_list.to_vec());
            make_large_geo_polygon(planet_model, std::slice::from_ref(&description)).map(Some)
        }
    }
}

/// [`tile`], with an exception a shape method raised (see
/// [`super::errors`]) propagating as Java's would.
fn catch_tile(
    planet_model: &Arc<PlanetModel>,
    pts: &[GeoPoint],
    filtered_point_list: &[usize],
    holes: &Option<Vec<Arc<dyn GeoPolygon>>>,
) -> TileResult<Arc<dyn GeoPolygon>> {
    catch(|| tile(planet_model, pts, filtered_point_list, holes))
        .unwrap_or_else(|e| Err(Fail::Err(e)))
}

/// The `try` block both `makeGeoPolygon`s share: find a point known to be
/// inside or outside, then tile.
fn tile(
    planet_model: &Arc<PlanetModel>,
    pts: &[GeoPoint],
    filtered_point_list: &[usize],
    holes: &Option<Vec<Arc<dyn GeoPolygon>>>,
) -> TileResult<Arc<dyn GeoPolygon>> {
    let filtered: Vec<GeoPoint> = filtered_point_list
        .iter()
        .map(|&i| pts[i].clone())
        .collect();
    // First approximation to find a point
    let center_of_mass = get_center_of_mass(planet_model, &filtered);
    if let Some(is_center_of_mass_inside) = is_inside_polygon(&center_of_mass, &filtered) {
        return generate_geo_polygon(
            planet_model,
            pts,
            filtered_point_list,
            holes,
            &center_of_mass,
            is_center_of_mass_inside,
        );
    }
    // Create a random number generator. Effectively this furnishes us with
    // a repeatable sequence of points to use for poles.
    let mut generator = JavaRandom::new(1234);
    for _ in 0..1_000_000 {
        let pole = pick_pole(&mut generator, planet_model, &filtered);
        if let Some(is_pole_inside) = is_inside_polygon(&pole, &filtered) {
            return generate_geo_polygon(
                planet_model,
                pts,
                filtered_point_list,
                holes,
                &pole,
                is_pole_inside,
            );
        }
    }
    Err(Fail::Err(Error::IllegalArgument(format!(
        "cannot find a point that is inside the polygon {}",
        points_string(&filtered)
    ))))
}

/// `getCenterOfMass(planetModel, points)`.
fn get_center_of_mass(planet_model: &PlanetModel, points: &[GeoPoint]) -> GeoPoint {
    let mut x = 0.0;
    let mut y = 0.0;
    let mut z = 0.0;
    for point in points {
        x += point.x;
        y += point.y;
        z += point.z;
    }
    // Normalization is not needed because createSurfacePoint does the
    // scaling anyway.
    planet_model.create_surface_point_xyz(x, y, z)
}

/// `makeLargeGeoPolygon(planetModel, shapesList)`: one
/// [`GeoComplexPolygon`] for all the rings and their holes.
pub fn make_large_geo_polygon(
    planet_model: &Arc<PlanetModel>,
    shapes_list: &[PolygonDescription],
) -> Result<Arc<dyn GeoPolygon>> {
    let mut points_list: Vec<Vec<GeoPoint>> = Vec::new();
    let mut test_point_shape: Option<BestShape> = None;
    for shape in shapes_list {
        test_point_shape = convert_polygon(&mut points_list, shape, test_point_shape, true)?;
    }
    let Some(test_point_shape) = test_point_shape else {
        return Err(illegal(
            "couldn't find a non-degenerate polygon for in-set determination",
        ));
    };
    let center_of_mass = get_center_of_mass(planet_model, &test_point_shape.points);
    if let Some(rval) =
        test_point_shape.create_geo_complex_polygon(planet_model, &points_list, &center_of_mass)?
    {
        return Ok(Arc::new(rval));
    }
    // Center of mass didn't work.
    let mut generator = JavaRandom::new(1234);
    for _ in 0..1_000_000 {
        let pole = pick_pole(&mut generator, planet_model, &test_point_shape.points);
        if let Some(rval) =
            test_point_shape.create_geo_complex_polygon(planet_model, &points_list, &pole)?
        {
            return Ok(Arc::new(rval));
        }
    }
    // Java concatenates the BestShape's default `Object.toString()`, an
    // identity hash; the class name is all that is reproducible.
    Err(Error::IllegalArgument(
        "cannot find a point that is inside the polygon org.apache.lucene.spatial3d.geom.GeoPolygonFactory$BestShape"
            .into(),
    ))
}

/// `convertPolygon(pointsList, shape, testPointShape, mustBeInside)`.
fn convert_polygon(
    points_list: &mut Vec<Vec<GeoPoint>>,
    shape: &PolygonDescription,
    mut test_point_shape: Option<BestShape>,
    must_be_inside: bool,
) -> Result<Option<BestShape>> {
    // First, remove duplicate points. If degenerate, just ignore the shape.
    let Some(filtered) = filter_points(&shape.points)? else {
        return Ok(test_point_shape);
    };
    let filtered_points: Vec<GeoPoint> =
        filtered.iter().map(|&i| shape.points[i].clone()).collect();
    // Non-degenerate. Check if this is a candidate for in-set determination.
    if shape.holes.is_empty()
        && test_point_shape
            .as_ref()
            .is_none_or(|t| t.points.len() > filtered_points.len())
    {
        test_point_shape = Some(BestShape {
            points: filtered_points.clone(),
            pole_must_be_inside: must_be_inside,
        });
    }
    points_list.push(filtered_points);
    // Now, do all holes too
    for hole in &shape.holes {
        test_point_shape = convert_polygon(points_list, hole, test_point_shape, !must_be_inside)?;
    }
    Ok(test_point_shape)
}

/// `BestShape`: the ring a test point is chosen against.
struct BestShape {
    points: Vec<GeoPoint>,
    pole_must_be_inside: bool,
}

impl BestShape {
    /// `createGeoComplexPolygon(planetModel, pointsList, testPoint)`:
    /// `None` when the test point is unusable.
    fn create_geo_complex_polygon(
        &self,
        planet_model: &Arc<PlanetModel>,
        points_list: &[Vec<GeoPoint>],
        test_point: &GeoPoint,
    ) -> Result<Option<GeoComplexPolygon>> {
        let Some(is_test_point_inside) = is_inside_polygon(test_point, &self.points) else {
            return Ok(None);
        };
        let built = if is_test_point_inside == self.pole_must_be_inside {
            GeoComplexPolygon::new(
                planet_model,
                points_list.to_vec(),
                test_point.clone(),
                is_test_point_inside,
            )
        } else {
            GeoComplexPolygon::new(
                planet_model,
                points_list.to_vec(),
                GeoPoint::new(-test_point.x, -test_point.y, -test_point.z),
                !is_test_point_inside,
            )
        };
        match built {
            Ok(p) => Ok(Some(p)),
            // Probably bad choice of test point.
            Err(Error::IllegalArgument(_)) => Ok(None),
            Err(e) => Err(e),
        }
    }
}

/// `generateGeoPolygon(planetModel, filteredPointList, holes, testPoint,
/// testPointInside)`.
fn generate_geo_polygon(
    planet_model: &Arc<PlanetModel>,
    pts: &[GeoPoint],
    filtered_point_list: &[usize],
    holes: &Option<Vec<Arc<dyn GeoPolygon>>>,
    test_point: &GeoPoint,
    test_point_inside: bool,
) -> TileResult<Arc<dyn GeoPolygon>> {
    // We will be trying twice to find the right GeoPolygon, using alternate
    // siding choices for the first polygon side.
    let initial_plane = SidedPlane::from_vectors(
        test_point,
        &pts[filtered_point_list[0]],
        &pts[filtered_point_list[1]],
    );
    let initial_plane = initial_plane?;
    let ctx = Tiler {
        planet_model,
        pts,
        holes: holes.as_deref(),
    };
    let mut rval = GeoCompositePolygon::new(planet_model);
    let mut seen_concave = false;
    let built = ctx.build_polygon_shape(
        &mut rval,
        &mut seen_concave,
        filtered_point_list,
        &[],
        0,
        1,
        initial_plane,
        Some(test_point),
    );
    let built = built?;
    // `false`: the test point was within the shape. Build it for real if
    // that was intended, else build the complement; `true`: keep what was
    // built if the point was meant to be outside, else the complement.
    if built && !test_point_inside {
        return Ok(Arc::new(rval));
    }
    let plane = if !built && test_point_inside {
        initial_plane
    } else {
        SidedPlane::opposite(&initial_plane)
    };
    let mut rval = GeoCompositePolygon::new(planet_model);
    let mut seen_concave = false;
    let built = ctx.build_polygon_shape(
        &mut rval,
        &mut seen_concave,
        filtered_point_list,
        &[],
        0,
        1,
        plane,
        None,
    );
    built?;
    Ok(Arc::new(rval))
}

/// `getLegalIndex(index, size)`.
fn legal_index(index: isize, size: usize) -> usize {
    index.rem_euclid(size as isize) as usize
}

/// `filterPoints(input)`: the ring without consecutive numerically
/// identical points (as indices into `input`), or `None` when fewer than
/// three remain.
fn filter_points(input: &[GeoPoint]) -> Result<Option<Vec<usize>>> {
    let n = input.len();
    let Some(compare_point) = input.first() else {
        return Err(Error::IndexOutOfBounds(
            "Index 0 out of bounds for length 0".into(),
        ));
    };
    // Backtrack to find something different from the first point
    let mut start_index = None;
    for i in 0..n.saturating_sub(1) {
        let the_point = &input[legal_index(-(i as isize) - 1, n)];
        if !the_point.is_numerically_identical(compare_point) {
            start_index = Some(legal_index(-(i as isize), n));
            break;
        }
    }
    let Some(start_index) = start_index else {
        return Ok(None);
    };
    // Now we can start the process of walking around, removing duplicate
    // points.
    let mut no_identical_points = Vec::with_capacity(n);
    let mut current_index = start_index;
    loop {
        let current_point = &input[current_index];
        no_identical_points.push(current_index);
        loop {
            current_index = legal_index(current_index as isize + 1, n);
            if current_index == start_index {
                break;
            }
            if !input[current_index].is_numerically_identical(current_point) {
                break;
            }
        }
        if current_index == start_index {
            break;
        }
    }
    if no_identical_points.len() < 3 {
        return Ok(None);
    }
    Ok(Some(no_identical_points))
}

/// `filterEdges(noIdenticalPoints, leniencyValue)`: the ring without
/// coplanar runs, or `None` when everything is coplanar.
fn filter_edges(
    pts: &[GeoPoint],
    no_identical_points: &[usize],
    _leniency_value: f64,
) -> Result<Option<Vec<usize>>> {
    for i in 0..no_identical_points.len() {
        // Search starting for current index.
        if let Some(result_path) = find_safe_path(pts, no_identical_points, i)? {
            if result_path.len() > 1 {
                return Ok(Some(result_path));
            }
        }
    }
    // No path found. This means that everything was coplanar.
    Ok(None)
}

/// `findSafePath(points, startIndex, leniencyValue)`: the `SafePath` chain's
/// points in order (`fillInList`), or `None`.
fn find_safe_path(
    pts: &[GeoPoint],
    points: &[usize],
    start_index: usize,
) -> Result<Option<Vec<usize>>> {
    let n = points.len();
    let mut safe_path: Option<Vec<usize>> = None;
    let mut i = start_index;
    while i < start_index + n {
        let start_point_index = legal_index(i as isize - 1, n);
        let start_point = &pts[points[start_point_index]];
        let mut end_point_index = legal_index(i as isize, n);
        let mut end_point = &pts[points[end_point_index]];
        if start_point.is_numerically_identical(end_point) {
            // Skip to next if end point is numerically identical to start
            // point.
            i += 1;
            continue;
        }
        // We have two points. Now look for a third that is not coplanar.
        loop {
            let next_point_index = legal_index(end_point_index as isize + 1, n);
            let next_point = &pts[points[next_point_index]];
            if start_point.is_numerically_identical(next_point) {
                // Degenerate: everything is the same point.
                return Ok(None);
            }
            if !Plane::are_points_coplanar(start_point, end_point, next_point)? {
                break;
            }
            if end_point_index == start_index {
                // We've looped around; everything is coplanar.
                return Ok(None);
            }
            end_point_index = next_point_index;
            end_point = next_point;
            i += 1;
        }
        if safe_path.is_some() && end_point_index == start_index {
            break;
        }
        // Java builds `new Plane(startPoint, endPoint)` for the path and
        // never reads it; constructing it can still throw.
        Plane::from_vectors(start_point, end_point)?;
        safe_path
            .get_or_insert_with(Vec::new)
            .push(points[end_point_index]);
        i += 1;
    }
    Ok(safe_path)
}

/// `pickPole(generator, planetModel, points)`: a random point near one of
/// the ring's points.
fn pick_pole(
    generator: &mut JavaRandom,
    planet_model: &PlanetModel,
    points: &[GeoPoint],
) -> GeoPoint {
    let point_index = generator.next_int_bounded(points.len() as i32) as usize;
    let close_point = &points[point_index];
    // We pick a random angle and random arc distance, then generate a point
    // based on closePoint
    let angle = generator.next_double() * PI * 2.0 - PI;
    let mut max_arc_distance = points[0].arc_distance(&points[1]);
    let trial_arc_distance = points[0].arc_distance(&points[2]);
    if trial_arc_distance > max_arc_distance {
        max_arc_distance = trial_arc_distance;
    }
    let arc_distance = max_arc_distance - generator.next_double() * max_arc_distance;
    // We come up with a unit circle (x,y,z) coordinate given the random
    // angle and arc distance. The point is centered around the positive x
    // axis.
    let x = cos(arc_distance);
    let sin_arc_distance = sin(arc_distance);
    let y = cos(angle) * sin_arc_distance;
    let z = sin(angle) * sin_arc_distance;
    // Now, use closePoint for a rotation pole
    let sin_latitude = sin(close_point.latitude());
    let cos_latitude = cos(close_point.latitude());
    let sin_longitude = sin(close_point.longitude());
    let cos_longitude = cos(close_point.longitude());
    // This transformation should take the point (1,0,0) and transform it to
    // the closepoint's actual (x,y,z) coordinates. Coordinate rotation
    // formula:
    let x1 = x * cos_latitude - z * sin_latitude;
    let y1 = y;
    let z1 = x * sin_latitude + z * cos_latitude;
    let x2 = x1 * cos_longitude - y1 * sin_longitude;
    let y2 = x1 * sin_longitude + y1 * cos_longitude;
    let z2 = z1;
    // Finally, scale to put the point on the surface
    planet_model.create_surface_point_xyz(x2, y2, z2)
}

/// `isInsidePolygon(point, polyPoints)`: whether the ring winds around
/// `point`, `None` when that cannot be decided.
fn is_inside_polygon(point: &GeoPoint, poly_points: &[GeoPoint]) -> Option<bool> {
    // First, compute sine and cosine of pole point latitude and longitude
    let latitude = point.latitude();
    let longitude = point.longitude();
    let sin_latitude = sin(latitude);
    let cos_latitude = cos(latitude);
    let sin_longitude = sin(longitude);
    let cos_longitude = cos(longitude);
    // Now, compute the incremental arc distance around the points of the
    // polygon
    let mut arc_distance = 0.0;
    let mut prev_angle: Option<f64> = None;
    let add = |angle: f64, prev_angle: f64| -> Option<f64> {
        let mut angle_delta = angle - prev_angle;
        if angle_delta < -PI {
            angle_delta += PI * 2.0;
        }
        if angle_delta > PI {
            angle_delta -= PI * 2.0;
        }
        if abs(angle_delta - PI) < MINIMUM_ANGULAR_RESOLUTION {
            return None;
        }
        Some(angle_delta)
    };
    for poly_point in poly_points {
        let angle = compute_angle(
            poly_point,
            sin_latitude,
            cos_latitude,
            sin_longitude,
            cos_longitude,
        );
        let angle = angle?;
        if let Some(prev) = prev_angle {
            arc_distance += add(angle, prev)?;
        }
        prev_angle = Some(angle);
    }
    // Do the final edge
    if let Some(prev) = prev_angle {
        let last_angle = compute_angle(
            &poly_points[0],
            sin_latitude,
            cos_latitude,
            sin_longitude,
            cos_longitude,
        );
        let last_angle = last_angle?;
        arc_distance += add(last_angle, prev)?;
    }
    // Clockwise == inside == negative
    if abs(arc_distance) < MINIMUM_ANGULAR_RESOLUTION {
        return None;
    }
    Some(arc_distance > 0.0)
}

/// `computeAngle(point, sinLatitude, cosLatitude, sinLongitude,
/// cosLongitude)`: the point's angle around the pole, `None` at the pole.
fn compute_angle(
    point: &GeoPoint,
    sin_latitude: f64,
    cos_latitude: f64,
    sin_longitude: f64,
    cos_longitude: f64,
) -> Option<f64> {
    // Coordinate rotation formula:
    // Rotate about the Z axis by longitude
    let x1 = point.x * cos_longitude + point.y * sin_longitude;
    let y1 = -point.x * sin_longitude + point.y * cos_longitude;
    let z1 = point.z;
    // Rotate about the Y axis by latitude
    let y2 = y1;
    let z2 = -x1 * sin_latitude + z1 * cos_latitude;
    // Now we use y and z for the angle computation
    if sqrt(y2 * y2 + z2 * z2) < MINIMUM_RESOLUTION {
        return None;
    }
    Some(atan2(z2, y2))
}

/// `Edge`: two points (indices into the caller's list) and the sided plane
/// through them.
#[derive(Debug, Clone)]
struct Edge {
    start_point: usize,
    end_point: usize,
    plane: SidedPlane,
    is_internal: bool,
}

/// `EdgeBuffer`: a ring of edges with previous/next links. Edges are never
/// moved; a removed edge loses its links (Java removes its map entries).
struct EdgeBuffer {
    edges: Vec<Edge>,
    alive: Vec<bool>,
    previous: Vec<Option<usize>>,
    next: Vec<Option<usize>>,
    size: usize,
    one_edge: Option<usize>,
}

impl EdgeBuffer {
    fn new(
        pts: &[GeoPoint],
        point_list: &[usize],
        internal_edges: &[bool],
        start_plane_start_index: usize,
        start_plane_end_index: usize,
        start_plane: SidedPlane,
    ) -> Result<EdgeBuffer> {
        let mut buffer = EdgeBuffer {
            edges: Vec::with_capacity(point_list.len()),
            alive: Vec::with_capacity(point_list.len()),
            previous: Vec::with_capacity(point_list.len()),
            next: Vec::with_capacity(point_list.len()),
            size: 0,
            one_edge: None,
        };
        let start_edge = buffer.push(Edge {
            start_point: point_list[start_plane_start_index],
            end_point: point_list[start_plane_end_index],
            plane: start_plane,
            is_internal: bit(internal_edges, start_plane_start_index),
        });
        let mut current_edge = start_edge;
        let mut end_index = start_plane_end_index;
        loop {
            if buffer.edges[current_edge].end_point == buffer.edges[start_edge].start_point {
                // We're done!
                buffer.previous[start_edge] = Some(current_edge);
                buffer.next[current_edge] = Some(start_edge);
                buffer.add_alive(start_edge);
                break;
            }
            let start_index = end_index;
            end_index += 1;
            if end_index >= point_list.len() {
                end_index -= point_list.len();
            }
            // Get the next point
            let new_point = &pts[point_list[end_index]];
            // Build the new edge
            let current = &buffer.edges[current_edge];
            let is_new_point_within = current.plane.is_within(new_point);
            let point_to_present = &pts[current.start_point];
            let new_plane = SidedPlane::from_vectors_on_side(
                point_to_present,
                is_new_point_within,
                &pts[point_list[start_index]],
                new_point,
            );
            let new_plane = new_plane?;
            let new_edge = buffer.push(Edge {
                start_point: point_list[start_index],
                end_point: point_list[end_index],
                plane: new_plane,
                is_internal: bit(internal_edges, start_index),
            });
            // Link it in
            buffer.previous[new_edge] = Some(current_edge);
            buffer.next[current_edge] = Some(new_edge);
            buffer.add_alive(new_edge);
            current_edge = new_edge;
        }
        buffer.one_edge = Some(start_edge);
        Ok(buffer)
    }

    fn push(&mut self, edge: Edge) -> usize {
        self.edges.push(edge);
        self.alive.push(false);
        self.previous.push(None);
        self.next.push(None);
        self.edges.len() - 1
    }

    fn add_alive(&mut self, e: usize) {
        if !self.alive[e] {
            self.alive[e] = true;
            self.size += 1;
        }
    }

    /// `getPrevious(edge)`; Java's map lookup is `null` for an unlinked
    /// edge, which no caller reaches.
    fn get_previous(&self, e: usize) -> usize {
        self.previous[e].expect("edge buffer link")
    }

    /// `getNext(edge)`.
    fn get_next(&self, e: usize) -> usize {
        self.next[e].expect("edge buffer link")
    }

    /// `replace(removeList, newEdge)`.
    fn replace(&mut self, remove_list: &[usize], new_edge: Edge) {
        let previous = self.previous[remove_list[0]];
        let next = self.next[remove_list[remove_list.len() - 1]];
        let new_edge = self.push(new_edge);
        self.add_alive(new_edge);
        self.previous[new_edge] = previous;
        if let Some(p) = previous {
            self.next[p] = Some(new_edge);
        }
        if let Some(n) = next {
            self.previous[n] = Some(new_edge);
        }
        self.next[new_edge] = next;
        for &edge in remove_list {
            if Some(edge) == self.one_edge {
                self.one_edge = Some(new_edge);
            }
            if self.alive[edge] {
                self.alive[edge] = false;
                self.size -= 1;
            }
            self.previous[edge] = None;
            self.next[edge] = None;
        }
    }

    /// `clear()`.
    fn clear(&mut self) {
        for e in 0..self.edges.len() {
            self.alive[e] = false;
            self.previous[e] = None;
            self.next[e] = None;
        }
        self.size = 0;
        self.one_edge = None;
    }

    /// `iterator()`: from `pickOne()` around the ring once (a snapshot; no
    /// caller changes the buffer while iterating).
    fn iter(&self) -> Vec<usize> {
        let mut rval = Vec::with_capacity(self.size);
        let Some(first_edge) = self.one_edge else {
            return rval;
        };
        let mut current = Some(first_edge);
        while let Some(c) = current {
            rval.push(c);
            current = self.next[c].filter(|&n| n != first_edge);
        }
        rval
    }
}

/// `BitSet.get(i)` on a growable flag vector.
fn bit(bits: &[bool], i: usize) -> bool {
    bits.get(i).copied().unwrap_or(false)
}

/// `BitSet.set(i, value)`.
fn set_bit(bits: &mut Vec<bool>, i: usize, value: bool) {
    if i >= bits.len() {
        if !value {
            return;
        }
        bits.resize(i + 1, false);
    }
    bits[i] = value;
}

/// The arguments every recursion of the tiler shares.
struct Tiler<'a> {
    planet_model: &'a Arc<PlanetModel>,
    pts: &'a [GeoPoint],
    holes: Option<&'a [Arc<dyn GeoPolygon>]>,
}

impl Tiler<'_> {
    fn holes_vec(&self) -> Option<Vec<Arc<dyn GeoPolygon>>> {
        self.holes.map(|h| h.to_vec())
    }

    fn has_holes(&self) -> bool {
        self.holes.is_some_and(|h| !h.is_empty())
    }

    fn values(&self, points: &[usize]) -> Vec<GeoPoint> {
        points.iter().map(|&i| self.pts[i].clone()).collect()
    }

    /// `buildPolygonShape(...)`: `false` when the test point turned out to
    /// be inside what was built.
    #[allow(clippy::too_many_arguments)]
    fn build_polygon_shape(
        &self,
        rval: &mut GeoCompositePolygon,
        seen_concave: &mut bool,
        points_list: &[usize],
        internal_edges: &[bool],
        start_point_index: usize,
        end_point_index: usize,
        starting_edge: SidedPlane,
        test_point: Option<&GeoPoint>,
    ) -> TileResult<bool> {
        let pts = self.pts;
        // Create the edge buffer.
        let edge_buffer = EdgeBuffer::new(
            pts,
            points_list,
            internal_edges,
            start_point_index,
            end_point_index,
            starting_edge,
        );
        let mut edge_buffer = edge_buffer?;
        // Starting state:
        // The stopping point
        let mut stopping_point = edge_buffer.one_edge;
        let mut current_edge = stopping_point;
        // Progressively look for convex sections. If we find one, we emit it
        // and replace it.
        while let Some(current) = current_edge {
            // Find convexity around the current edge, if any
            let Some(found_it) =
                self.find_convex_polygon(current, rval, &mut edge_buffer, test_point)?
            else {
                return Ok(false);
            };
            if found_it {
                // New start point
                stopping_point = edge_buffer.one_edge;
                current_edge = stopping_point;
                continue;
            }
            // Otherwise, go on to the next
            let next = edge_buffer.get_next(current);
            if Some(next) == stopping_point {
                break;
            }
            current_edge = Some(next);
        }
        // Look for any reason that the concave polygon cannot be created.
        let mut found_bad_edge = false;
        for check_edge in edge_buffer.iter() {
            let check = &edge_buffer.edges[check_edge];
            let flipped_plane = SidedPlane::opposite(&check.plane);
            // Now walk around again looking for points that fail.
            for confirm_edge in edge_buffer.iter() {
                if confirm_edge == check_edge {
                    continue;
                }
                let confirm = &edge_buffer.edges[confirm_edge];
                // Look for a point that is on the wrong side of the check
                // edge. This means that we can't build the polygon.
                let the_point = if check.start_point != confirm.start_point
                    && check.end_point != confirm.start_point
                    && !flipped_plane.is_within(&pts[confirm.start_point])
                {
                    Some(confirm.start_point)
                } else if check.start_point != confirm.end_point
                    && check.end_point != confirm.end_point
                    && !flipped_plane.is_within(&pts[confirm.end_point])
                {
                    Some(confirm.end_point)
                } else {
                    None
                };
                let Some(the_point) = the_point else {
                    continue;
                };
                // Note that we found a problem.
                found_bad_edge = true;
                // Check the edge: is it coplanar with the point? If so,
                // continue.
                let coplanar = Plane::are_points_coplanar(
                    &pts[check.start_point],
                    &pts[check.end_point],
                    &pts[the_point],
                );
                if coplanar? {
                    continue;
                }
                // Build and add the convex triangle, then split.
                let third_part_points = vec![check.start_point, check.end_point, the_point];
                let mut third_part_internal = Vec::new();
                set_bit(&mut third_part_internal, 0, check.is_internal);
                set_bit(&mut third_part_internal, 1, true);
                let convex_part = GeoConvexPolygon::new(
                    self.planet_model,
                    self.values(&third_part_points),
                    self.holes_vec(),
                    third_part_internal,
                    true,
                );
                let convex_part = convex_part?;
                rval.add_shape(Arc::new(convex_part))?;
                // The part preceding the bad edge, back to the point
                let mut loop_edge = edge_buffer.get_previous(check_edge);
                let mut first_part_points = Vec::new();
                let mut first_part_internal = Vec::new();
                let mut i = 0;
                loop {
                    let e = &edge_buffer.edges[loop_edge];
                    first_part_points.push(e.end_point);
                    if e.end_point == the_point {
                        break;
                    }
                    set_bit(&mut first_part_internal, i, e.is_internal);
                    i += 1;
                    loop_edge = edge_buffer.get_previous(loop_edge);
                }
                set_bit(&mut first_part_internal, i, true);
                let first_plane = SidedPlane::from_vectors_on_side(
                    &pts[check.end_point],
                    false,
                    &pts[check.start_point],
                    &pts[the_point],
                );
                let first_plane = first_plane?;
                let built = self.build_polygon_shape(
                    rval,
                    seen_concave,
                    &first_part_points,
                    &first_part_internal,
                    first_part_points.len() - 1,
                    0,
                    first_plane,
                    test_point,
                );
                if !built? {
                    return Ok(false);
                }
                // The part following the bad edge, forward to the point
                let mut second_part_points = Vec::new();
                let mut second_part_internal = Vec::new();
                loop_edge = edge_buffer.get_next(check_edge);
                i = 0;
                loop {
                    let e = &edge_buffer.edges[loop_edge];
                    second_part_points.push(e.start_point);
                    if e.start_point == the_point {
                        break;
                    }
                    set_bit(&mut second_part_internal, i, e.is_internal);
                    i += 1;
                    loop_edge = edge_buffer.get_next(loop_edge);
                }
                set_bit(&mut second_part_internal, i, true);
                let second_plane = SidedPlane::from_vectors_on_side(
                    &pts[check.start_point],
                    false,
                    &pts[check.end_point],
                    &pts[the_point],
                );
                let second_plane = second_plane?;
                let built = self.build_polygon_shape(
                    rval,
                    seen_concave,
                    &second_part_points,
                    &second_part_internal,
                    second_part_points.len() - 1,
                    0,
                    second_plane,
                    test_point,
                );
                if !built? {
                    return Ok(false);
                }
                return Ok(true);
            }
        }
        if found_bad_edge {
            // Unaddressable coplanarity; give up.
            return Err(Fail::Tile(
                "Could not tile polygon; found a pathological coplanarity that couldn't be addressed".into(),
            ));
        }
        // No violations found: we know it's a legal concave polygon.
        self.make_concave_polygon(rval, seen_concave, &edge_buffer, test_point)
    }

    /// `makeConcavePolygon(...)`.
    fn make_concave_polygon(
        &self,
        rval: &mut GeoCompositePolygon,
        seen_concave: &mut bool,
        edge_buffer: &EdgeBuffer,
        test_point: Option<&GeoPoint>,
    ) -> TileResult<bool> {
        if edge_buffer.size == 0 {
            return Ok(true);
        }
        if *seen_concave {
            return Err(Fail::Err(illegal(
                "Illegal polygon; polygon edges intersect each other",
            )));
        }
        *seen_concave = true;
        // If there are less than three edges, something got messed up
        // somehow. Don't know how this can happen but check.
        if edge_buffer.size < 3 {
            return Err(Fail::Err(illegal(
                "Illegal polygon; polygon edges intersect each other",
            )));
        }
        // Create the list of points
        let mut points = Vec::with_capacity(edge_buffer.size);
        let mut internal_edges = Vec::new();
        let mut edge = edge_buffer.one_edge.expect("non-empty edge buffer");
        let mut is_internal = false;
        for i in 0..edge_buffer.size {
            let e = &edge_buffer.edges[edge];
            points.push(e.start_point);
            if i < edge_buffer.size - 1 {
                set_bit(&mut internal_edges, i, e.is_internal);
            } else {
                is_internal = e.is_internal;
            }
            edge = edge_buffer.get_next(edge);
        }
        self.emit(rval, points, internal_edges, is_internal, test_point, true)
            .map(|added| added.is_some())
    }

    /// The `try` that ends both `findConvexPolygon` and
    /// `makeConcavePolygon`: build the polygon (first without holes when
    /// there are holes and a test point), reject it when it contains the
    /// test point (`Ok(None)`), else add it (`Ok(Some(true))`). An
    /// `IllegalArgumentException` -- from the constructor or a hole's
    /// `isWithin` -- becomes a `TileException`.
    fn emit(
        &self,
        rval: &mut GeoCompositePolygon,
        points: Vec<usize>,
        internal_edges: Vec<bool>,
        return_is_internal: bool,
        test_point: Option<&GeoPoint>,
        concave: bool,
    ) -> TileResult<Option<bool>> {
        let pm = self.planet_model;
        let make = |holes: Option<Vec<Arc<dyn GeoPolygon>>>| -> Result<Arc<dyn GeoPolygon>> {
            let values = self.values(&points);
            let flags = internal_edges.clone();
            let built: Result<Arc<dyn GeoPolygon>> = if concave {
                GeoConcavePolygon::new(pm, values, holes, flags, return_is_internal)
                    .map(|p| Arc::new(p) as _)
            } else {
                GeoConvexPolygon::new(pm, values, holes, flags, return_is_internal)
                    .map(|p| Arc::new(p) as _)
            };
            built
        };
        let attempt = catch(|| -> Result<Option<Arc<dyn GeoPolygon>>> {
            if let (Some(tp), true) = (test_point, self.has_holes()) {
                let test_polygon = make(None)?;
                if test_polygon.is_within(tp) {
                    return Ok(None);
                }
            }
            let real_polygon = make(self.holes_vec())?;
            if let (Some(tp), false) = (test_point, self.has_holes()) {
                if real_polygon.is_within(tp) {
                    return Ok(None);
                }
            }
            Ok(Some(real_polygon))
        })
        .and_then(|r| r);
        match attempt {
            Ok(Some(real_polygon)) => {
                rval.add_shape(real_polygon)
                    .map_err(|e| Fail::Tile(e.to_string()))?;
                Ok(Some(true))
            }
            Ok(None) => Ok(None),
            Err(Error::IllegalArgument(msg)) => Err(Fail::Tile(msg)),
            Err(e) => Err(Fail::Err(e)),
        }
    }

    /// `findConvexPolygon(...)`: `Some(true)` when a convex polygon around
    /// `current_edge` was emitted, `Some(false)` when there is none, `None`
    /// when it would contain the test point.
    fn find_convex_polygon(
        &self,
        current_edge: usize,
        rval: &mut GeoCompositePolygon,
        edge_buffer: &mut EdgeBuffer,
        test_point: Option<&GeoPoint>,
    ) -> TileResult<Option<bool>> {
        let pts = self.pts;
        let parallel = || {
            Fail::Tile("Two adjacent edge planes are effectively parallel despite filtering; give up on tiling".into())
        };
        // Order of insertion does not matter: membership is all `contains`
        // and an all-of test.
        let mut included_edges: Vec<usize> = vec![current_edge];
        let mut first_edge = current_edge;
        let mut last_edge = current_edge;
        let eb = &*edge_buffer;
        let ed = |e: usize| &eb.edges[e];
        let sp = |e: usize| ed(e).start_point;
        let ep = |e: usize| ed(e).end_point;
        let cop =
            |a: usize, b: usize, c: usize| Plane::are_points_coplanar(&pts[a], &pts[b], &pts[c]);
        let sided =
            |p: usize, a: usize, b: usize| SidedPlane::from_vectors(&pts[p], &pts[a], &pts[b]);
        let within_all = |point: usize, included: &[usize]| {
            included.iter().all(|&e| ed(e).plane.is_within(&pts[point]))
        };
        let within_ext = |point: usize,
                          included: &[usize],
                          extension: usize,
                          return_boundary: &Option<SidedPlane>| {
            if !ed(extension).plane.is_within(&pts[point]) {
                return false;
            }
            if let Some(rb) = return_boundary {
                if !rb.is_within(&pts[point]) {
                    return false;
                }
            }
            within_all(point, included)
        };
        // Extend forward.
        loop {
            if ed(first_edge).start_point == ed(last_edge).end_point {
                break;
            }
            let new_last_edge = eb.get_next(last_edge);
            if cop(sp(last_edge), ep(last_edge), ep(new_last_edge))? {
                break;
            }
            if ed(last_edge)
                .plane
                .is_functionally_identical(&ed(new_last_edge).plane)
            {
                return Err(parallel());
            }
            if within_all(ed(new_last_edge).end_point, &included_edges) {
                let return_boundary = if ed(first_edge).start_point != ed(new_last_edge).end_point {
                    if cop(ep(first_edge), sp(first_edge), ep(new_last_edge))?
                        || cop(sp(first_edge), ep(new_last_edge), sp(new_last_edge))?
                    {
                        break;
                    }
                    Some(sided(ep(first_edge), sp(first_edge), ep(new_last_edge))?)
                } else {
                    None
                };
                let mut found_point_inside = false;
                for edge in eb.iter() {
                    if !included_edges.contains(&edge) && edge != new_last_edge {
                        if ed(edge).start_point != ed(new_last_edge).end_point
                            && within_ext(
                                ed(edge).start_point,
                                &included_edges,
                                new_last_edge,
                                &return_boundary,
                            )
                        {
                            found_point_inside = true;
                            break;
                        }
                        if ed(edge).end_point != ed(first_edge).start_point
                            && within_ext(
                                ed(edge).end_point,
                                &included_edges,
                                new_last_edge,
                                &return_boundary,
                            )
                        {
                            found_point_inside = true;
                            break;
                        }
                    }
                }
                if !found_point_inside {
                    included_edges.push(new_last_edge);
                    last_edge = new_last_edge;
                    continue;
                }
            }
            break;
        }
        // Extend backward.
        loop {
            if ed(first_edge).start_point == ed(last_edge).end_point {
                break;
            }
            let new_first_edge = eb.get_previous(first_edge);
            if cop(sp(new_first_edge), ep(new_first_edge), ep(first_edge))? {
                break;
            }
            if ed(first_edge)
                .plane
                .is_functionally_identical(&ed(new_first_edge).plane)
            {
                return Err(parallel());
            }
            if within_all(ed(new_first_edge).start_point, &included_edges) {
                let return_boundary = if ed(new_first_edge).start_point != ed(last_edge).end_point {
                    if cop(sp(last_edge), ep(last_edge), sp(new_first_edge))?
                        || cop(ep(last_edge), sp(new_first_edge), ep(new_first_edge))?
                    {
                        break;
                    }
                    Some(sided(sp(last_edge), ep(last_edge), sp(new_first_edge))?)
                } else {
                    None
                };
                let mut found_point_inside = false;
                for edge in eb.iter() {
                    if !included_edges.contains(&edge) && edge != new_first_edge {
                        if ed(edge).start_point != ed(last_edge).end_point
                            && within_ext(
                                ed(edge).start_point,
                                &included_edges,
                                new_first_edge,
                                &return_boundary,
                            )
                        {
                            found_point_inside = true;
                            break;
                        }
                        if ed(edge).end_point != ed(new_first_edge).start_point
                            && within_ext(
                                ed(edge).end_point,
                                &included_edges,
                                new_first_edge,
                                &return_boundary,
                            )
                        {
                            found_point_inside = true;
                            break;
                        }
                    }
                }
                if !found_point_inside {
                    included_edges.push(new_first_edge);
                    first_edge = new_first_edge;
                    continue;
                }
            }
            break;
        }
        // Ok, figure out what we've accumulated. If it is enough for a
        // polygon, build it.
        if included_edges.len() < 2 {
            return Ok(Some(false));
        }
        let mut points = Vec::with_capacity(included_edges.len() + 1);
        let mut internal_edges = Vec::new();
        let return_is_internal;
        if ed(first_edge).start_point == ed(last_edge).end_point {
            // We're going to convert the whole remaining ring.
            if included_edges.len() < 3 {
                // Not enough edges to make a polygon.
                return Ok(Some(false));
            }
            if ed(first_edge)
                .plane
                .is_functionally_identical(&ed(last_edge).plane)
            {
                return Err(parallel());
            }
            let mut edge = first_edge;
            points.push(ed(edge).start_point);
            let mut k = 0;
            loop {
                if edge == last_edge {
                    break;
                }
                points.push(ed(edge).end_point);
                set_bit(&mut internal_edges, k, ed(edge).is_internal);
                k += 1;
                edge = eb.get_next(edge);
            }
            return_is_internal = ed(last_edge).is_internal;
            edge_buffer.clear();
        } else {
            // Build the return edge (internal, of course)
            let return_sided_plane = SidedPlane::from_vectors_on_side(
                &pts[ed(first_edge).end_point],
                false,
                &pts[ed(first_edge).start_point],
                &pts[ed(last_edge).end_point],
            );
            let return_sided_plane = return_sided_plane?;
            let return_edge = Edge {
                start_point: ed(first_edge).start_point,
                end_point: ed(last_edge).end_point,
                plane: return_sided_plane,
                is_internal: true,
            };
            if return_edge
                .plane
                .is_functionally_identical(&ed(last_edge).plane)
                || return_edge
                    .plane
                    .is_functionally_identical(&ed(first_edge).plane)
            {
                return Err(parallel());
            }
            // Build point list and edge list
            let mut edges = Vec::with_capacity(included_edges.len());
            return_is_internal = true;
            let mut edge = first_edge;
            points.push(ed(edge).start_point);
            let mut k = 0;
            loop {
                points.push(ed(edge).end_point);
                set_bit(&mut internal_edges, k, ed(edge).is_internal);
                k += 1;
                edges.push(edge);
                if edge == last_edge {
                    break;
                }
                edge = eb.get_next(edge);
            }
            // Modify the edge buffer
            edge_buffer.replace(&edges, return_edge);
        }
        self.emit(
            rval,
            points,
            internal_edges,
            return_is_internal,
            test_point,
            false,
        )
    }
}
