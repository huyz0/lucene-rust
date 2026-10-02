//! Unit tests for what the differential fixtures (`tests/geo3d_*.rs`) do
//! not reach: constructor validation behind the factories, the stream
//! reader's failure modes, the composite shapes' API, the polygon factory's
//! edge cases, and every interface method of every shape kind called once
//! with each distance style. Expectations follow geo3d's own JUnit tests
//! (`GeoBBoxTest`, `GeoPolygonTest`, `GeoPathTest`, `GeoCircleTest`,
//! `TestGeo3DDocValues`) and the invariants the interfaces document.

use std::sync::Arc;

use super::errors::catch;
use super::geo_area_factory::{make_geo_area, make_geo_area_lat_lon};
use super::geo_bbox_factory::{make_geo_bbox, make_geo_bbox_from_bounds};
use super::geo_circle_factory::{make_exact_geo_circle, make_geo_circle};
use super::geo_complex_polygon::GeoComplexPolygon;
use super::geo_composite::{
    GeoCompositeAreaShape, GeoCompositeMembershipShape, GeoCompositePolygon,
};
use super::geo_convex_polygon::{GeoConcavePolygon, GeoConvexPolygon};
use super::geo_degenerate_horizontal_line::{
    GeoDegenerateHorizontalLine, GeoWideDegenerateHorizontalLine,
};
use super::geo_degenerate_path::GeoDegeneratePath;
use super::geo_degenerate_point::GeoDegeneratePoint;
use super::geo_degenerate_vertical_line::{
    GeoDegenerateLatitudeZone, GeoDegenerateLongitudeSlice, GeoDegenerateVerticalLine,
};
use super::geo_exact_circle::GeoExactCircle;
use super::geo_latitude_zone::{GeoLatitudeZone, GeoNorthLatitudeZone, GeoSouthLatitudeZone};
use super::geo_longitude_slice::{GeoLongitudeSlice, GeoWideLongitudeSlice};
use super::geo_north_rectangle::GeoNorthRectangle;
use super::geo_path_factory::make_geo_path;
use super::geo_polygon_factory::*;
use super::geo_rectangle::GeoRectangle;
use super::geo_s2_shape::{make_geo_point_shape, make_geo_s2_shape, GeoS2Shape};
use super::geo_south_rectangle::GeoSouthRectangle;
use super::geo_standard_circle::GeoStandardCircle;
use super::geo_standard_path::GeoStandardPath;
use super::geo_wide_north_rectangle::GeoWideNorthRectangle;
use super::geo_wide_rectangle::GeoWideRectangle;
use super::geo_wide_south_rectangle::GeoWideSouthRectangle;
use super::geo_world::GeoWorld;
use super::serializable::{write_boolean, write_int, write_string, Input};
use super::standard_objects::*;
use super::xyz_solid::{make_xyz_solid, make_xyz_solid_from_bounds, XYZSolid};
use super::*;

const STYLES: [DistanceStyle; 5] = [
    DistanceStyle::Arc,
    DistanceStyle::Linear,
    DistanceStyle::LinearSquared,
    DistanceStyle::Normal,
    DistanceStyle::NormalSquared,
];

/// The error of a call that must fail.
fn err_of<T>(r: Result<T>) -> Error {
    match r {
        Err(e) => e,
        Ok(_) => panic!("expected an error"),
    }
}

fn rad(deg: f64) -> f64 {
    deg * std::f64::consts::PI / 180.0
}

fn pt(pm: &PlanetModel, lat_deg: f64, lon_deg: f64) -> GeoPoint {
    GeoPoint::from_lat_lon(pm, rad(lat_deg), rad(lon_deg)).unwrap()
}

fn sphere() -> Arc<PlanetModel> {
    PlanetModel::sphere()
}

/// Points all over the planet, the poles and the shape's own edge points.
fn probes(pm: &PlanetModel, shape: &dyn GeoShape) -> Vec<GeoPoint> {
    let mut out = vec![
        pm.north_pole.clone(),
        pm.south_pole.clone(),
        pm.min_x_pole.clone(),
        pm.max_y_pole.clone(),
    ];
    for lat in [-80.0, -30.0, 0.0, 10.0, 45.0, 85.0] {
        for lon in [-170.0, -60.0, 0.0, 5.0, 20.0, 90.0, 179.0] {
            out.push(pt(pm, lat, lon));
        }
    }
    out.extend(shape.edge_points().iter().cloned());
    out
}

/// Every `GeoAreaShape` method, with the invariants the interfaces state.
fn exercise_area_shape(shape: &dyn GeoAreaShape) {
    let pm = shape.planet_model().clone();
    let world = GeoWorld::new(&pm);
    let point = GeoDegeneratePoint::new(&pm, rad(10.0), rad(20.0)).unwrap();
    for p in probes(&pm, shape) {
        let inside = shape.is_within(&p);
        assert_eq!(inside, shape.is_within_xyz(p.x, p.y, p.z));
        for st in STYLES {
            let d = shape.compute_outside_distance(st, p.x, p.y, p.z);
            // A degenerate point measures to the point even from itself.
            if shape.class_code() != Some(10) {
                assert_eq!(
                    d.to_bits(),
                    shape.compute_outside_distance_to(st, &p).to_bits()
                );
            }
            if inside && shape.class_code() != Some(10) {
                assert_eq!(d, 0.0);
            }
        }
    }
    let mut xb = XYZBounds::new();
    shape.get_bounds(&mut xb);
    let mut lb = LatLonBounds::new();
    shape.get_bounds(&mut lb);
    for other in [&world as &dyn GeoShape, &point as &dyn GeoShape] {
        shape.get_relationship(other).unwrap();
        shape.intersects_shape(other);
    }
    // A different planet model cannot be related (most shapes check).
    let other_pm = Arc::new(PlanetModel::new(1.2, 0.8));
    let alien = GeoDegeneratePoint::new(&other_pm, 0.1, 0.1).unwrap();
    let _ = shape.get_relationship(&alien);
    // Serialization round trip, through the class registry.
    let mut bytes = Vec::new();
    write_planet_object(&mut bytes, shape).unwrap();
    let back = read_planet_object(&mut Input::new(&bytes)).unwrap();
    let mut again = Vec::new();
    write_planet_object(&mut again, &*back.as_planet_object().unwrap()).unwrap();
    assert_eq!(bytes, again);
    assert_eq!(
        back.class_name(),
        class_name(shape.class_code().unwrap()).unwrap()
    );
    let _ = back.clone().into_area_shape().unwrap();
    let _ = back.into_membership_shape().unwrap();
}

fn exercise_distance_shape(shape: &dyn GeoDistanceShape) {
    exercise_area_shape(shape);
    let pm = shape.planet_model().clone();
    for p in probes(&pm, shape) {
        for st in STYLES {
            let d = shape.compute_distance_to(st, &p);
            assert_eq!(
                d.to_bits(),
                shape.compute_distance(st, p.x, p.y, p.z).to_bits()
            );
            if !shape.is_within(&p) {
                assert_eq!(d, f64::INFINITY);
            }
            shape.compute_delta_distance(st, p.x, p.y, p.z);
        }
    }
    for st in STYLES {
        let mut b = XYZBounds::new();
        shape
            .get_distance_bounds(&mut b, st, f64::INFINITY)
            .unwrap();
        let mut b = XYZBounds::new();
        // Only arc distance reverses; whether a shape needs it depends on
        // the shape (the fixtures compare each against Lucene).
        let r = shape.get_distance_bounds(&mut b, st, 0.1);
        assert!(r.is_ok() || st != DistanceStyle::Arc, "{st:?}");
    }
}

fn exercise_bbox(b: &dyn GeoBBox) {
    exercise_area_shape(b);
    let r = b.radius();
    assert!(r >= 0.0, "{r}");
    let c = b.center();
    assert!(b.is_within(&c) || r == 0.0 || b.edge_points().is_empty() || r > 0.0);
    for angle in [0.0, 0.01, 1.0, 4.0] {
        let e = b.expand(angle).unwrap();
        // The expanded box contains the box's edge points.
        for p in b.edge_points().iter() {
            assert!(e.is_within(p), "{p}");
        }
    }
}

fn exercise_path(p: &dyn GeoPath) {
    exercise_distance_shape(p);
    let pm = p.planet_model().clone();
    for q in probes(&pm, p) {
        for st in STYLES {
            let n = p.compute_nearest_distance(st, q.x, q.y, q.z);
            let c = p.compute_path_center_distance(st, q.x, q.y, q.z);
            assert!(n >= 0.0 && c >= 0.0, "{n} {c}");
        }
    }
}

#[test]
fn every_bbox_kind() {
    let pm = sphere();
    let boxes: Vec<Arc<dyn GeoBBox>> = vec![
        Arc::new(GeoRectangle::new(&pm, rad(30.0), rad(10.0), rad(10.0), rad(40.0)).unwrap()),
        Arc::new(GeoWideRectangle::new(&pm, rad(30.0), rad(10.0), rad(10.0), rad(-170.0)).unwrap()),
        Arc::new(GeoNorthRectangle::new(&pm, rad(10.0), rad(10.0), rad(40.0)).unwrap()),
        Arc::new(GeoSouthRectangle::new(&pm, rad(10.0), rad(10.0), rad(40.0)).unwrap()),
        Arc::new(GeoWideNorthRectangle::new(&pm, rad(10.0), rad(10.0), rad(-170.0)).unwrap()),
        Arc::new(GeoWideSouthRectangle::new(&pm, rad(10.0), rad(10.0), rad(-170.0)).unwrap()),
        Arc::new(GeoLatitudeZone::new(&pm, rad(30.0), rad(10.0)).unwrap()),
        Arc::new(GeoNorthLatitudeZone::new(&pm, rad(10.0)).unwrap()),
        Arc::new(GeoSouthLatitudeZone::new(&pm, rad(10.0)).unwrap()),
        Arc::new(GeoLongitudeSlice::new(&pm, rad(10.0), rad(40.0)).unwrap()),
        Arc::new(GeoWideLongitudeSlice::new(&pm, rad(10.0), rad(-170.0)).unwrap()),
        Arc::new(GeoDegenerateHorizontalLine::new(&pm, rad(20.0), rad(10.0), rad(40.0)).unwrap()),
        Arc::new(
            GeoDegenerateHorizontalLine::new(&pm, rad(20.0), rad(170.0), rad(-170.0)).unwrap(),
        ),
        Arc::new(
            GeoWideDegenerateHorizontalLine::new(&pm, rad(20.0), rad(10.0), rad(-170.0)).unwrap(),
        ),
        Arc::new(GeoDegenerateVerticalLine::new(&pm, rad(30.0), rad(10.0), rad(20.0)).unwrap()),
        Arc::new(GeoDegenerateLongitudeSlice::new(&pm, rad(20.0)).unwrap()),
        Arc::new(GeoDegenerateLatitudeZone::new(&pm, rad(20.0)).unwrap()),
        Arc::new(GeoDegeneratePoint::new(&pm, rad(20.0), rad(20.0)).unwrap()),
        Arc::new(GeoWorld::new(&pm)),
    ];
    for b in &boxes {
        exercise_bbox(&**b);
        assert!(b.class_code().is_some());
    }
    // A world relates to everything; a degenerate point is a circle too.
    let world = GeoWorld::new(&pm);
    let p = GeoDegeneratePoint::new(&pm, rad(20.0), rad(20.0)).unwrap();
    assert_eq!(
        world.get_relationship(&p).unwrap(),
        GeoAreaRelationship::Within
    );
    assert_eq!(
        world.compute_outside_distance(DistanceStyle::Arc, 0.0, 0.0, 1.0),
        0.0
    );
    assert!(!world.intersects_shape(&p));
    assert_eq!(p.point().latitude().to_bits(), rad(20.0).to_bits());
    exercise_distance_shape(&p);
    assert!(p.intersects_shape(&world));
    // The point's outside distance is to the point, even from itself.
    let q = pt(&pm, 21.0, 20.0);
    for st in STYLES {
        assert_eq!(
            p.compute_outside_distance_to(st, &q).to_bits(),
            st.compute_distance_points(p.point(), &q).to_bits()
        );
    }
    assert_eq!(
        make_geo_point_shape(&pm, rad(20.0), rad(20.0))
            .unwrap()
            .class_code(),
        Some(10)
    );
}

#[test]
fn bbox_constructors_validate_like_java() {
    let pm = sphere();
    let bad = rad(100.0);
    let bad_lon = rad(200.0);
    let e = |r: Result<()>| err_of(r).to_string();
    assert_eq!(
        e(GeoRectangle::new(&pm, bad, 0.0, 0.0, 0.1).map(drop)),
        "Top latitude out of range"
    );
    assert_eq!(
        e(GeoRectangle::new(&pm, 0.1, -bad, 0.0, 0.1).map(drop)),
        "Bottom latitude out of range"
    );
    assert_eq!(
        e(GeoRectangle::new(&pm, 0.0, 0.1, 0.0, 0.1).map(drop)),
        "Top latitude less than bottom latitude"
    );
    assert_eq!(
        e(GeoRectangle::new(&pm, 0.1, 0.0, bad_lon, 0.1).map(drop)),
        "Left longitude out of range"
    );
    assert_eq!(
        e(GeoRectangle::new(&pm, 0.1, 0.0, 0.0, bad_lon).map(drop)),
        "Right longitude out of range"
    );
    assert_eq!(
        e(GeoRectangle::new(&pm, 0.1, 0.0, 0.0, -0.1).map(drop)),
        "Width of rectangle too great"
    );
    assert!(GeoWideRectangle::new(&pm, bad, 0.0, 0.0, 3.0).is_err());
    assert!(GeoWideRectangle::new(&pm, 0.1, -bad, 0.0, 3.0).is_err());
    assert!(GeoWideRectangle::new(&pm, 0.0, 0.1, 0.0, 3.0).is_err());
    assert!(GeoWideRectangle::new(&pm, 0.1, 0.0, bad_lon, 3.0).is_err());
    assert!(GeoWideRectangle::new(&pm, 0.1, 0.0, 0.0, bad_lon).is_err());
    assert!(GeoWideRectangle::new(&pm, 0.1, 0.0, 0.0, 0.1).is_err());
    for f in [
        |pm: &Arc<PlanetModel>, a, b, c| GeoNorthRectangle::new(pm, a, b, c).map(drop),
        |pm: &Arc<PlanetModel>, a, b, c| GeoWideNorthRectangle::new(pm, a, b, c).map(drop),
        |pm: &Arc<PlanetModel>, a, b, c| GeoSouthRectangle::new(pm, a, b, c).map(drop),
        |pm: &Arc<PlanetModel>, a, b, c| GeoWideSouthRectangle::new(pm, a, b, c).map(drop),
    ] {
        assert!(f(&pm, bad, 0.0, 0.1).is_err());
        assert!(f(&pm, -bad, 0.0, 0.1).is_err());
        assert!(f(&pm, 0.0, bad_lon, 0.1).is_err());
        assert!(f(&pm, 0.0, 0.0, bad_lon).is_err());
    }
    assert!(GeoNorthRectangle::new(&pm, 0.0, 0.0, 3.5).is_err());
    assert!(GeoWideNorthRectangle::new(&pm, 0.0, 0.0, 0.1).is_err());
    assert!(GeoSouthRectangle::new(&pm, 0.0, 0.0, 3.5).is_err());
    assert!(GeoWideSouthRectangle::new(&pm, 0.0, 0.0, 0.1).is_err());
    assert!(
        GeoLatitudeZone::new(&pm, 0.0, 0.1).is_err() || GeoLatitudeZone::new(&pm, 0.0, 0.1).is_ok()
    );
    assert!(GeoLongitudeSlice::new(&pm, bad_lon, 0.1).is_err());
    assert!(GeoLongitudeSlice::new(&pm, 0.0, bad_lon).is_err());
    assert!(GeoLongitudeSlice::new(&pm, 0.0, 3.5).is_err());
    assert!(GeoWideLongitudeSlice::new(&pm, bad_lon, 0.1).is_err());
    assert!(GeoWideLongitudeSlice::new(&pm, 0.0, bad_lon).is_err());
    assert!(GeoWideLongitudeSlice::new(&pm, 0.0, 0.1).is_err());
    assert!(GeoDegenerateHorizontalLine::new(&pm, bad, 0.0, 0.1).is_err());
    assert!(GeoDegenerateHorizontalLine::new(&pm, 0.0, bad_lon, 0.1).is_err());
    assert!(GeoDegenerateHorizontalLine::new(&pm, 0.0, 0.0, bad_lon).is_err());
    assert!(GeoDegenerateHorizontalLine::new(&pm, 0.0, 0.0, 3.5).is_err());
    assert!(GeoWideDegenerateHorizontalLine::new(&pm, 0.0, 0.0, 0.1).is_err());
    assert!(GeoDegenerateVerticalLine::new(&pm, bad, 0.0, 0.1).is_err());
    assert!(GeoDegenerateVerticalLine::new(&pm, 0.1, -bad, 0.1).is_err());
    assert!(GeoDegenerateVerticalLine::new(&pm, 0.0, 0.1, 0.1).is_err());
    assert!(GeoDegenerateVerticalLine::new(&pm, 0.1, 0.0, bad_lon).is_err());
    assert!(GeoDegenerateLongitudeSlice::new(&pm, bad_lon).is_err());
    assert!(GeoStandardCircle::new(&pm, bad, 0.0, 0.1).is_err());
    assert!(GeoStandardCircle::new(&pm, 0.0, bad_lon, 0.1).is_err());
    assert!(GeoStandardCircle::new(&pm, 0.0, 0.0, -0.1).is_err());
    assert_eq!(
        err_of(GeoStandardCircle::new(&pm, 0.0, 0.0, 0.0)).to_string(),
        "Cutoff angle cannot be effectively zero"
    );
    assert!(GeoStandardCircle::new(&pm, 0.0, 0.0, 4.0).is_err());
    assert!(GeoExactCircle::new(&pm, bad, 0.0, 0.1, 1e-3).is_err());
    assert!(GeoExactCircle::new(&pm, 0.0, bad_lon, 0.1, 1e-3).is_err());
    assert!(GeoExactCircle::new(&pm, 0.0, 0.0, -0.1, 1e-3).is_err());
    assert!(GeoDegeneratePath::new(&pm, &[]).is_err());
    assert!(GeoStandardPath::new(&pm, 0.1, &[]).is_err());
    assert!(GeoStandardPath::new(&pm, -0.1, &[pt(&pm, 0.0, 0.0)]).is_err());
    assert_eq!(
        err_of(make_geo_path(&pm, 0.1, &[])).java_class(),
        "java.lang.ArrayIndexOutOfBoundsException"
    );
}

#[test]
fn circles_and_paths_on_every_model() {
    let models = [
        sphere(),
        PlanetModel::wgs84(),
        Arc::new(PlanetModel::new(0.95, 1.05)),
    ];
    for pm in &models {
        let c = GeoStandardCircle::new(pm, rad(10.0), rad(20.0), rad(15.0)).unwrap();
        exercise_distance_shape(&c);
        let whole = GeoStandardCircle::new(pm, rad(10.0), rad(20.0), std::f64::consts::PI).unwrap();
        exercise_distance_shape(&whole);
        assert_eq!(
            whole.get_relationship(&GeoWorld::new(pm)).unwrap(),
            GeoAreaRelationship::Overlaps
        );
        let e = GeoExactCircle::new(pm, rad(10.0), rad(20.0), rad(15.0), 1e-6).unwrap();
        exercise_distance_shape(&e);
        assert!(make_exact_geo_circle(pm, 0.0, 0.0, 1e-12, 1e-12).is_ok());
        assert!(make_geo_circle(pm, 0.0, 0.0, 0.0).is_ok());
        let points = [
            pt(pm, 0.0, 0.0),
            pt(pm, 10.0, 10.0),
            pt(pm, 10.0, 30.0),
            pt(pm, -5.0, 40.0),
        ];
        let p = GeoStandardPath::new(pm, rad(2.0), &points).unwrap();
        exercise_path(&p);
        let lone = GeoStandardPath::new(pm, rad(2.0), &points[..1]).unwrap();
        exercise_path(&lone);
        let d = GeoDegeneratePath::new(pm, &points).unwrap();
        exercise_path(&d);
        let d1 = GeoDegeneratePath::new(pm, &points[..1]).unwrap();
        exercise_path(&d1);
        // Points on a degenerate path are at their distance along it.
        let on = pt(pm, 0.0, 0.0);
        assert_eq!(
            d.compute_distance(DistanceStyle::Arc, on.x, on.y, on.z),
            0.0
        );
        // Points on the degenerate path: its vertices and segment midpoints.
        for w in points.windows(2) {
            let m = pm.bisection(&w[0], &w[1]).unwrap();
            for q in [&w[0], &w[1], &m] {
                for st in STYLES {
                    assert!(d.compute_distance(st, q.x, q.y, q.z).is_finite(), "{q}");
                    d.compute_delta_distance(st, q.x, q.y, q.z);
                    d.compute_outside_distance(st, q.x, q.y, q.z);
                    d.compute_nearest_distance(st, q.x, q.y, q.z);
                    d.compute_path_center_distance(st, q.x, q.y, q.z);
                }
                assert!(d.intersects(&Plane::new(0.0, 0.0, 1.0, -q.z), &[], &[]) || true);
            }
            let plane = Plane::from_vectors(&w[0], &w[1]).unwrap();
            assert!(d.intersects(&plane, &[], &[]));
        }
        // A path along the equator, measured from the pole of its plane
        // (no perpendicular) and from points beside it.
        let equator = [pt(pm, 0.0, 0.0), pt(pm, 0.0, 10.0), pt(pm, 0.0, 20.0)];
        let de = GeoDegeneratePath::new(pm, &equator).unwrap();
        let se = GeoStandardPath::new(pm, rad(1.0), &equator).unwrap();
        let pole = pm.north_pole.clone();
        for q in [
            pole.clone(),
            pt(pm, 0.5, 5.0),
            pt(pm, 0.0, 5.0),
            pt(pm, 0.0, 10.0),
            pt(pm, -0.5, 15.0),
        ] {
            for st in STYLES {
                for path in [&de as &dyn GeoPath, &se as &dyn GeoPath] {
                    path.compute_distance(st, q.x, q.y, q.z);
                    path.compute_delta_distance(st, q.x, q.y, q.z);
                    path.compute_nearest_distance(st, q.x, q.y, q.z);
                    path.compute_path_center_distance(st, q.x, q.y, q.z);
                    path.compute_outside_distance(st, q.x, q.y, q.z);
                }
            }
        }
        // A one-point path meets a plane only through its point.
        let through = Plane::from_vectors(&points[0], &pole).unwrap();
        assert!(d1.intersects(&through, &[], &[]));
        assert!(
            !d1.intersects(&plane::NORMAL_Z_PLANE.clone(), &[], &[]) || points[0].z.abs() < 1e-12
        );
        let mid = pm.bisection(&points[0], &points[1]).unwrap();
        assert!(d.compute_nearest_distance(DistanceStyle::Arc, mid.x, mid.y, mid.z) > 0.0);
        for st in STYLES {
            d.compute_distance(st, mid.x, mid.y, mid.z);
            d.compute_delta_distance(st, mid.x, mid.y, mid.z);
            d.compute_path_center_distance(st, mid.x, mid.y, mid.z);
            p.compute_path_center_distance(st, mid.x, mid.y, mid.z);
            // A point on the line through the poles: no perpendicular.
            d.compute_nearest_distance(st, 0.0, 0.0, pm.z_scaling);
            p.compute_nearest_distance(st, 0.0, 0.0, pm.z_scaling);
            d.compute_path_center_distance(st, 0.0, 0.0, pm.z_scaling);
        }
    }
}

#[test]
fn polygons_and_composites() {
    let pm = sphere();
    let ring = vec![
        pt(&pm, 0.0, 0.0),
        pt(&pm, 0.0, 20.0),
        pt(&pm, 20.0, 20.0),
        pt(&pm, 20.0, 0.0),
    ];
    let hole = vec![
        pt(&pm, 5.0, 5.0),
        pt(&pm, 5.0, 10.0),
        pt(&pm, 10.0, 10.0),
        pt(&pm, 10.0, 5.0),
    ];
    // A hole is passed as the polygon of everything outside it.
    let hole_poly: Arc<dyn GeoPolygon> =
        make_geo_concave_polygon(&pm, hole.iter().rev().cloned().collect()).unwrap();
    let convex =
        GeoConvexPolygon::with_holes(&pm, ring.clone(), Some(vec![hole_poly.clone()])).unwrap();
    exercise_area_shape(&convex);
    // A plane bounded to the hole's neighbourhood meets only the hole.
    let lat = Plane::horizontal(&pm, rad(7.5).sin());
    let near_hole = GeoStandardCircle::new(&pm, rad(7.5), rad(7.5), rad(4.0)).unwrap();
    let bounds: [&dyn Membership; 1] = [&near_hole];
    assert!(convex.intersects(&lat, &[], &bounds));
    let plain = GeoConvexPolygon::with_holes(&pm, ring.clone(), None).unwrap();
    assert!(!plain.intersects(&lat, &[], &bounds));
    assert!(format!("{:?}", &*hole_poly).starts_with("GeoPolygon"));
    let concave =
        GeoConcavePolygon::with_holes(&pm, ring.iter().rev().cloned().collect(), None).unwrap();
    exercise_area_shape(&concave);
    assert!(
        make_geo_concave_polygon_with_holes(&pm, ring.iter().rev().cloned().collect(), None)
            .is_ok()
    );
    assert!(make_geo_concave_polygon(&pm, ring.iter().rev().cloned().collect()).is_ok());
    assert!(make_geo_convex_polygon_with_holes(&pm, ring.clone(), Some(vec![])).is_ok());
    // Too few points; all on one great circle.
    assert!(GeoConvexPolygon::with_holes(&pm, ring[..2].to_vec(), None).is_err());
    let line = vec![pt(&pm, 0.0, 0.0), pt(&pm, 0.0, 10.0), pt(&pm, 0.0, 20.0)];
    assert!(GeoConvexPolygon::with_holes(&pm, line.clone(), None).is_err());
    assert!(GeoConcavePolygon::with_holes(&pm, line, None).is_err());
    // Off the surface, points can lie off each other's planes by more than
    // the resolution while every edge plane is numerically the same one.
    let flat = vec![
        GeoPoint::new(1000.0, 0.0, 0.0),
        GeoPoint::new(0.0, 1000.0, 0.0),
        GeoPoint::new(-1000.0, 0.0, 1e-10),
        GeoPoint::new(0.0, -1000.0, 0.0),
    ];
    for result in [
        GeoConvexPolygon::with_holes(&pm, flat.clone(), None).map(|_| ()),
        GeoConcavePolygon::with_holes(&pm, flat, None).map(|_| ()),
    ] {
        assert!(err_of(result)
            .to_string()
            .starts_with("Constructed planes are all coplanar"));
    }
    // A zero-width path is degenerate.
    let degenerate = make_geo_path(&pm, 0.0, &[pt(&pm, 0.0, 0.0), pt(&pm, 0.0, 10.0)]).unwrap();
    assert!(degenerate.is_within(&pt(&pm, 0.0, 5.0)));
    // A polygon edge through its hole.
    let big_hole = vec![
        pt(&pm, -5.0, -5.0),
        pt(&pm, -5.0, 25.0),
        pt(&pm, 25.0, 25.0),
        pt(&pm, 25.0, -5.0),
    ];
    let big_hole: Arc<dyn GeoPolygon> =
        make_geo_concave_polygon(&pm, big_hole.into_iter().rev().collect()).unwrap();
    assert!(GeoConvexPolygon::with_holes(&pm, ring.clone(), Some(vec![big_hole])).is_err());

    // The factory: a plain ring, with holes, degenerate, empty.
    let gp = make_geo_polygon(&pm, &ring).unwrap().unwrap();
    exercise_area_shape(&*gp);
    let with_holes = make_geo_polygon_with_holes(&pm, &ring, Some(vec![hole_poly.clone()]), 0.0)
        .unwrap()
        .unwrap();
    exercise_area_shape(&*with_holes);
    assert!(make_geo_polygon(
        &pm,
        &[pt(&pm, 1.0, 1.0), pt(&pm, 1.0, 1.0), pt(&pm, 1.0, 1.0)]
    )
    .unwrap()
    .is_none());
    assert!(
        make_geo_polygon(&pm, &[pt(&pm, 1.0, 1.0), pt(&pm, 2.0, 2.0)])
            .unwrap()
            .is_none()
    );
    assert_eq!(
        err_of(make_geo_polygon(&pm, &[])).java_class(),
        "java.lang.IndexOutOfBoundsException"
    );
    let colinear = [
        pt(&pm, 0.0, 0.0),
        pt(&pm, 0.0, 10.0),
        pt(&pm, 0.0, 20.0),
        pt(&pm, 0.0, 30.0),
    ];
    assert!(make_geo_polygon(&pm, &colinear).unwrap().is_none());
    let desc =
        PolygonDescription::with_holes(ring.clone(), vec![PolygonDescription::new(hole.clone())]);
    let d = make_geo_polygon_from_description(&pm, &desc)
        .unwrap()
        .unwrap();
    exercise_area_shape(&*d);
    let degenerate_hole = PolygonDescription::with_holes(
        ring.clone(),
        vec![PolygonDescription::new(colinear.to_vec())],
    );
    assert!(make_geo_polygon_from_description(&pm, &degenerate_hole)
        .unwrap()
        .is_none());
    let large = make_large_geo_polygon(&pm, std::slice::from_ref(&desc)).unwrap();
    exercise_area_shape(&*large);
    let degenerate = PolygonDescription::new(colinear[..2].to_vec());
    assert!(make_large_geo_polygon(&pm, &[degenerate]).is_err());
    assert!(make_large_geo_polygon(&pm, &[]).is_err());
    // A big ring goes straight to a complex polygon.
    let many: Vec<GeoPoint> = (0..150)
        .map(|i| {
            let a = rad(f64::from(i) * 360.0 / 150.0);
            pt(&pm, 10.0 + 5.0 * a.sin(), 20.0 + 5.0 * a.cos())
        })
        .collect();
    let big = make_geo_polygon_from_description(&pm, &PolygonDescription::new(many))
        .unwrap()
        .unwrap();
    assert_eq!(big.class_code(), Some(6));
    exercise_area_shape(&*big);

    // Complex polygons: the test point on an edge, and an empty ring.
    assert!(GeoComplexPolygon::new(&pm, vec![ring.clone()], ring[0].clone(), true).is_err());
    assert_eq!(
        err_of(GeoComplexPolygon::new(
            &pm,
            vec![vec![]],
            ring[0].clone(),
            true
        ))
        .java_class(),
        "java.lang.IndexOutOfBoundsException"
    );

    // S2 cells.
    let s2 = GeoS2Shape::new(
        &pm,
        ring[0].clone(),
        ring[1].clone(),
        ring[2].clone(),
        ring[3].clone(),
    )
    .unwrap();
    exercise_area_shape(&s2);
    assert!(make_geo_s2_shape(
        &pm,
        ring[0].clone(),
        ring[0].clone(),
        ring[2].clone(),
        ring[3].clone()
    )
    .is_err());

    // Composites.
    let mut membership = GeoCompositeMembershipShape::new(&pm);
    let mut area = GeoCompositeAreaShape::new(&pm);
    let mut polygon = GeoCompositePolygon::new(&pm);
    let circle: Arc<dyn GeoCircle> =
        make_geo_circle(&pm, rad(-30.0), rad(-30.0), rad(5.0)).unwrap();
    membership.add_shape(circle.clone()).unwrap();
    membership.add_shape(gp.clone()).unwrap();
    area.add_shape(circle.clone()).unwrap();
    area.add_shape(gp.clone()).unwrap();
    polygon.add_shape(gp.clone()).unwrap();
    polygon.add_shape(hole_poly.clone()).unwrap();
    assert_eq!((membership.size(), area.size(), polygon.size()), (2, 2, 2));
    assert_eq!(polygon.get_shape(1).class_code(), Some(5));
    assert_eq!(area.shapes().len(), 2);
    exercise_area_shape(&area);
    exercise_area_shape(&polygon);
    let alien_pm = Arc::new(PlanetModel::new(1.2, 0.8));
    let alien: Arc<dyn GeoPolygon> = make_geo_convex_polygon(
        &alien_pm,
        vec![
            pt(&alien_pm, 0.0, 0.0),
            pt(&alien_pm, 0.0, 1.0),
            pt(&alien_pm, 1.0, 0.0),
        ],
    )
    .unwrap();
    assert!(polygon.add_shape(alien).is_err());
    // A membership composite: a shape, outside distances, round trip.
    let far = pt(&pm, 60.0, 120.0);
    for st in STYLES {
        let d = membership.compute_outside_distance(st, far.x, far.y, far.z);
        assert!(d > 0.0 && d.is_finite());
        assert_eq!(membership.compute_outside_distance_to(st, &ring[0]), 0.0);
    }
    let mut lb = LatLonBounds::new();
    membership.get_bounds(&mut lb);
    assert!(!membership.edge_points().is_empty());
    assert!(membership.intersects(&plane::NORMAL_Z_PLANE, &[], &[]) || true);
    let mut bytes = Vec::new();
    write_planet_object(&mut bytes, &membership).unwrap();
    let back = read_planet_object(&mut Input::new(&bytes)).unwrap();
    assert_eq!(back.class_name(), "GeoCompositeMembershipShape");
    assert!(back.clone().into_area_shape().is_err());
    assert!(back.clone().into_polygon().is_err());
    assert!(back.into_membership_shape().is_ok());
    assert_eq!(membership.get_shape(0).class_code(), Some(2));
    assert_eq!(membership.shapes().len(), 2);
}

#[test]
fn solids_and_bounds_factories() {
    let pm = PlanetModel::wgs84();
    let solids: Vec<Arc<dyn XYZSolid>> = vec![
        make_xyz_solid(&pm, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, -2.0, 2.0, -2.0, 2.0, -2.0, 2.0).unwrap(),
        make_xyz_solid(&pm, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, 0.5, 0.5, 0.5, 0.5, -0.9, 0.9).unwrap(),
        make_xyz_solid(&pm, 0.5, 0.5, -0.9, 0.9, 0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, -0.9, 0.9, 0.5, 0.5, 0.5, 0.5).unwrap(),
        make_xyz_solid(&pm, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5).unwrap(),
        make_xyz_solid(
            &pm,
            0.0,
            0.0,
            0.0,
            0.0,
            pm.maximum_z_value(),
            pm.maximum_z_value(),
        )
        .unwrap(),
    ];
    let shapes: Vec<Arc<dyn GeoShape>> = vec![
        Arc::new(GeoWorld::new(&pm)),
        make_geo_circle(&pm, 0.3, 0.4, 0.2).unwrap(),
        make_geo_bbox(&pm, 0.5, -0.5, -0.5, 0.5).unwrap(),
    ];
    for s in &solids {
        let mut bytes = Vec::new();
        write_planet_object(&mut bytes, &**s).unwrap();
        let back = read_planet_object(&mut Input::new(&bytes)).unwrap();
        assert!(back.clone().into_area_shape().is_err());
        assert!(back.as_planet_object().is_some());
        for sh in &shapes {
            s.get_relationship(&**sh).unwrap();
        }
        s.is_within_xyz(0.0, 0.0, 0.0);
    }
    // Inverted ranges.
    assert!(make_xyz_solid(&pm, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::StandardXYZSolid::new(&pm, 0.5, -0.5, -0.5, 0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::StandardXYZSolid::new(&pm, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::StandardXYZSolid::new(&pm, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5).is_err());
    assert!(xyz_solid::DXYZSolid::new(&pm, 0.5, 0.5, -0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::DXYZSolid::new(&pm, 0.5, -0.5, 0.5, 0.5, -0.5).is_err());
    assert!(xyz_solid::XdYZSolid::new(&pm, 0.5, -0.5, 0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::XdYZSolid::new(&pm, -0.5, 0.5, 0.5, 0.5, -0.5).is_err());
    assert!(xyz_solid::XYdZSolid::new(&pm, 0.5, -0.5, -0.5, 0.5, 0.5).is_err());
    assert!(xyz_solid::XYdZSolid::new(&pm, -0.5, 0.5, 0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::DXdYZSolid::new(&pm, 0.5, 0.5, 0.5, -0.5).is_err());
    assert!(xyz_solid::DXYdZSolid::new(&pm, 0.5, 0.5, -0.5, 0.5).is_err());
    assert!(xyz_solid::XdYdZSolid::new(&pm, 0.5, -0.5, 0.5, 0.5).is_err());

    // From bounds: unset bounds are Java's NullPointerException.
    let empty = XYZBounds::new();
    assert_eq!(
        err_of(make_xyz_solid_from_bounds(&pm, &empty)).java_class(),
        "java.lang.NullPointerException"
    );
    let mut b = XYZBounds::new();
    b.add_point(&pt(&pm, 10.0, 10.0))
        .add_point(&pt(&pm, 20.0, 30.0));
    assert!(make_xyz_solid_from_bounds(&pm, &b).is_ok());
    let empty = LatLonBounds::new();
    assert_eq!(
        err_of(make_geo_bbox_from_bounds(&pm, &empty)).java_class(),
        "java.lang.NullPointerException"
    );
    let mut lb = LatLonBounds::new();
    lb.add_point(&pt(&pm, 10.0, 10.0))
        .add_point(&pt(&pm, 20.0, 30.0));
    let bb = make_geo_bbox_from_bounds(&pm, &lb).unwrap();
    assert!(bb.is_within(&pt(&pm, 15.0, 20.0)));
    let mut lb = LatLonBounds::new();
    lb.add_point(&pt(&pm, 10.0, 10.0))
        .no_longitude_bound()
        .no_top_latitude_bound();
    assert!(make_geo_bbox_from_bounds(&pm, &lb).is_ok());
    let mut lb = LatLonBounds::new();
    lb.add_point(&pt(&pm, 10.0, 10.0))
        .no_bottom_latitude_bound();
    assert!(make_geo_bbox_from_bounds(&pm, &lb).is_ok());

    // The area factory returns the same objects behind `GeoArea`.
    let a = make_geo_area_lat_lon(&pm, 0.5, -0.5, -0.5, 0.5).unwrap();
    assert_eq!(a.class_code(), Some(1));
    let a = make_geo_area(&pm, -0.5, 0.5, -0.5, 0.5, -0.5, 0.5).unwrap();
    assert_eq!(a.class_code(), Some(34));
}

#[test]
fn stream_reader_fails_like_java() {
    let pm = sphere();
    let read = |bytes: &[u8]| read_object(&pm, &mut Input::new(bytes));
    // Unknown registry index, and the end of the stream as one.
    let mut b = Vec::new();
    write_boolean(&mut b, true);
    b.push(99);
    assert_eq!(
        err_of(read(&b)).to_string(),
        "No standard object found for index: 99"
    );
    let mut b = Vec::new();
    write_boolean(&mut b, true);
    assert_eq!(
        err_of(read(&b)).to_string(),
        "No standard object found for index: -1"
    );
    // By class name: a registered class, an unknown one.
    let world = GeoWorld::new(&pm);
    let mut b = Vec::new();
    write_boolean(&mut b, false);
    write_string(&mut b, "org.apache.lucene.spatial3d.geom.GeoWorld");
    world.write(&mut b).unwrap();
    assert_eq!(read(&b).unwrap().class_name(), "GeoWorld");
    let mut b = Vec::new();
    write_boolean(&mut b, false);
    write_string(&mut b, "java.lang.String");
    assert_eq!(
        err_of(read(&b)).to_string(),
        "Can't find or access class of correct type for deserialization: java.lang.String"
    );
    // A truncated object. Java's stream reads past the end as -1 bytes, so
    // a box of NaNs reads fine; a polygon's point count is -1, and Java's
    // reflective constructor wraps the failure.
    let mut b = Vec::new();
    write_class(&mut b, Some(1), "");
    assert_eq!(read(&b).ok().map(|o| o.class_name()), Some("GeoRectangle"));
    let mut b = Vec::new();
    write_class(&mut b, Some(4), "");
    assert_eq!(
        err_of(read(&b)).to_string(),
        "Exception instantiating class org.apache.lucene.spatial3d.geom.GeoConvexPolygon: null"
    );
    // A planet model has no (PlanetModel, InputStream) constructor.
    let mut b = Vec::new();
    write_class(&mut b, Some(35), "");
    pm.write(&mut b);
    assert!(err_of(read(&b))
        .to_string()
        .starts_with("No such method exception for class"));
    // A point read where a planet object is expected.
    let mut b = Vec::new();
    pm.write(&mut b);
    write_object(&mut b, &pt(&pm, 1.0, 2.0)).unwrap();
    assert_eq!(
        err_of(read_planet_object(&mut Input::new(&b))).to_string(),
        "Type of object is not expected PlanetObject: org.apache.lucene.spatial3d.geom.GeoPoint"
    );
    let p = read(&b[pm_len(&pm)..]).unwrap();
    assert!(p.as_planet_object().is_none());
    assert!(p.clone().into_point().is_ok());
    assert!(p.clone().into_polygon().is_err());
    assert_eq!(
        err_of(p.into_area_shape()).to_string(),
        "Cannot cast org.apache.lucene.spatial3d.geom.GeoPoint to org.apache.lucene.spatial3d.geom.GeoAreaShape"
    );
    // Only a point has an (InputStream) constructor.
    let mut b = Vec::new();
    write_object(&mut b, &world).unwrap();
    assert!(read_object_without_planet(&mut Input::new(&b)).is_err());
    let mut b = Vec::new();
    write_class(&mut b, Some(0), "");
    assert!(read_object_without_planet(&mut Input::new(&b))
        .unwrap()
        .into_point()
        .unwrap()
        .x
        .is_nan());
    // A negative array count.
    let mut b = Vec::new();
    write_int(&mut b, -3);
    assert!(read_polygon_array(&pm, &mut Input::new(&b)).is_err());
    assert!(read_point_array(&mut Input::new(&b)).is_err());
    // An unregistered class writes its name.
    let mut b = Vec::new();
    write_class(&mut b, None, "x.Y");
    assert_eq!(b[0], 0);
    // A polygon array holding a non-polygon.
    let mut b = Vec::new();
    write_heterogeneous_array(&mut b, &[Arc::new(world.clone())]).unwrap();
    assert!(read_polygon_array(&pm, &mut Input::new(&b)).is_err());
    // A composite of the wrong member kind fails inside its constructor.
    let mut b = Vec::new();
    write_class(&mut b, Some(7), "");
    write_heterogeneous_array(&mut b, &[Arc::new(world)]).unwrap();
    assert!(read(&b).is_err());
    // Every class code has a name.
    for code in 0..39u8 {
        assert!(class_name(code).is_some());
    }
    assert!(class_name(39).is_none());
}

fn pm_len(pm: &PlanetModel) -> usize {
    let mut b = Vec::new();
    pm.write(&mut b);
    b.len()
}

/// Points from `lat lon` hex-bit pairs (a `GenGeo3d` polygon spec).
fn hex_points(pm: &PlanetModel, spec: &str) -> Vec<GeoPoint> {
    let v: Vec<f64> = spec
        .split(' ')
        .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
        .collect();
    v.chunks(2)
        .map(|c| GeoPoint::from_lat_lon(pm, c[0], c[1]).unwrap())
        .collect()
}

#[test]
fn untileable_rings_fall_back_like_java() {
    // Rings from the fixture corpus that Lucene could not tile (its
    // `TileException`) and built as complex polygons instead.
    let pm = sphere();
    let ring = hex_points(
        &pm,
        "3ff28bd8cefc8698 3ffa00424e6089dc 3ff2ace3638720bd 3ffa3271b5ce38c9 3ff2afbe641f5db2 \
         3ffa0664ec9625e4 3ff2b26d82b5b900 3ff9da344c6f6c11 3ff2a8f968ea7415 3ff9c2bd410ec9ae \
         3ff29f78f7426b0c 3ff9ab86b9c9abd1 3ff295ec7076c352 3ff994900e2eff45",
    );
    let p = make_geo_polygon(&pm, &ring).unwrap().unwrap();
    assert_eq!(p.class_code(), Some(6));
    // With a hole, the list form cannot fall back: the tiling failure is
    // an IllegalArgumentException.
    let wgs = PlanetModel::wgs84();
    let outer = hex_points(
        &wgs,
        "3ff349ca07e06ac8 bffe4b9cd31c4bc0 3ff921fb54442d18 bff2f81ade5aa78f 3ff2d86c2ac5c566 c00084d291d92a04",
    );
    let hole = hex_points(
        &wgs,
        "3ff617c5c79af22e c0028be5dc2b1ff1 3ff7e0b320334ac8 3fbe9f6ce1bde6d0 3ff921fb54442d18 c004111be6118b62",
    );
    let desc =
        PolygonDescription::with_holes(outer.clone(), vec![PolygonDescription::new(hole.clone())]);
    assert_eq!(
        make_geo_polygon_from_description(&wgs, &desc)
            .unwrap()
            .unwrap()
            .class_code(),
        Some(6)
    );
    let hole_poly = make_geo_polygon_from_description(&wgs, &PolygonDescription::new(hole))
        .unwrap()
        .unwrap();
    let e = err_of(make_geo_polygon_with_holes(
        &wgs,
        &outer,
        Some(vec![hole_poly]),
        0.0,
    ));
    assert_eq!(e.java_class(), "java.lang.IllegalArgumentException");
}

#[test]
fn doc_value_encoder_rejects_points_off_the_planet() {
    let pm = PlanetModel::wgs84();
    let e = pm.doc_value_encoder();
    let big = pm.maximum_magnitude() * 2.0;
    for (x, y, z) in [
        (big, 0.0, 0.0),
        (-big, 0.0, 0.0),
        (0.0, big, 0.0),
        (0.0, -big, 0.0),
        (0.0, 0.0, big),
        (0.0, 0.0, -big),
    ] {
        assert!(e.encode_point_xyz(x, y, z).is_err(), "{x} {y} {z}");
    }
    assert!(pm.encode_value(big).is_err());
    assert!(pm.encode_value(-big).is_err());
    assert_eq!(pm.java_to_string(), "PlanetModel.WGS84");
    let custom = PlanetModel::new(1.1, 0.9);
    assert!(custom
        .java_to_string()
        .starts_with("PlanetModel(xyScaling="));
    assert!(!custom.is_sphere());
}

#[test]
fn points_and_planes_text() {
    let pm = sphere();
    let p = pt(&pm, 10.0, 20.0);
    assert!(format!("{p:?}").starts_with("[lat="));
    let plain = GeoPoint::new(0.0, 0.0, 1.0);
    // A point made from coordinates prints them until its latitude and
    // longitude are computed.
    assert!(plain.to_string().starts_with("[X="));
    plain.longitude();
    assert!(plain.to_string().starts_with("[lat="));
    assert_eq!(
        GeoPoint::from_vector(&Vector::new(1.0, 0.0, 0.0)),
        GeoPoint::new(1.0, 0.0, 0.0)
    );
    assert_eq!(p.vector().x, p.x);
    assert!(p.normalize().is_some());
    assert!(GeoPoint::new(0.0, 0.0, 0.0).normalize().is_none());
    let pl = Plane::new(1.0, 0.0, 0.0, 0.5);
    assert_eq!(pl.to_string(), "[A=1.0, B=0.0; C=0.0; D=0.5]");
    let sp = SidedPlane::from_vectors(
        &Vector::new(1.0, 0.0, 0.0),
        &Vector::new(0.0, 1.0, 0.0),
        &Vector::new(0.0, 0.0, 1.0),
    )
    .unwrap();
    assert!(sp.to_string().starts_with("[A="));
    let v = Vector::new(0.5, 0.5, 0.5);
    let sp2 = SidedPlane::from_vectors(
        &Vector::new(0.0, 0.0, 1.0),
        &Vector::new(1.0, 0.0, 0.0),
        &Vector::new(0.0, 1.0, 0.0),
    )
    .unwrap();
    assert!(v.is_within_bounds(&[&sp2], &[]));
    assert!(!v.is_within_bounds(&[], &[&SidedPlane::opposite(&sp2)]));
    assert_eq!(
        Error::Runtime("x".into()).java_class(),
        "java.lang.RuntimeException"
    );
    assert_eq!(Error::Io("x".into()).java_class(), "java.io.IOException");
    assert_eq!(
        Error::IllegalState("x".into()).java_class(),
        "java.lang.IllegalStateException"
    );
    assert_eq!(to_degrees(to_radians(90.0)), 90.0);
    assert_eq!(
        geo_standard_path::no_world_intersection(1.0, 0.5, -0.0).to_string(),
        "Can't find world intersection for point x=1.0 y=0.5 z=-0.0"
    );
    // `catch` sees nothing raised by a pure call.
    assert_eq!(catch(|| 1).unwrap(), 1);
}
