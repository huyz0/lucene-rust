//! Unit tests for the Spatial4j subset: configuration, error and edge paths
//! the differential fixtures (`tests/spatial4j_fixtures.rs`) do not reach.

use std::collections::BTreeMap;
use std::sync::Arc;

use super::binary_codec::{write_double, DataInput};
use super::wkt::{java_is_whitespace, java_parse_double};
use super::*;

fn args(kv: &[(&str, &str)]) -> BTreeMap<String, String> {
    kv.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn geo() -> Arc<SpatialContext> {
    SpatialContext::geo_context()
}

fn cart() -> Arc<SpatialContext> {
    SpatialContextFactory::make_spatial_context(&args(&[("geo", "false")])).unwrap()
}

#[test]
fn context_factory_settings_and_errors() {
    let ctx = SpatialContextFactory::make_spatial_context(&args(&[
        ("geo", "false"),
        ("distCalculator", "cartesian^2"),
        ("worldBounds", "ENVELOPE(-10, 10, 20, -20)"),
        (
            "shapeFactoryClass",
            "org.locationtech.spatial4j.shape.impl.ShapeFactoryImpl",
        ),
        (
            "binaryCodecClass",
            "org.locationtech.spatial4j.io.BinaryCodec",
        ),
    ]))
    .unwrap();
    assert!(!ctx.is_geo());
    assert_eq!(ctx.world_bounds_values(), [-10.0, 10.0, -20.0, 20.0]);
    assert_eq!(
        ctx.to_string(),
        "SpatialContext{geo=false, calculator=CartesianDistCalc, worldBounds=Rect(minX=-10.0,maxX=10.0,minY=-20.0,maxY=20.0)}"
    );
    assert!(format!("{ctx:?}").starts_with("SpatialContext{"));
    let p = ctx.point_xy(3.0, 4.0).unwrap();
    assert_eq!(ctx.calc_distance(&*p, 0.0, 0.0).unwrap(), 25.0);
    for (k, v, msg) in [
        ("distCalculator", "nope", "Unknown calculator: nope"),
        (
            "spatialContextFactory",
            "com.example.Nope",
            "java.lang.ClassNotFoundException: com.example.Nope",
        ),
        (
            "shapeFactoryClass",
            "com.example.Nope",
            "Invalid value 'com.example.Nope' on field shapeFactoryClass of type class java.lang.Class",
        ),
    ] {
        let e = SpatialContextFactory::make_spatial_context(&args(&[(k, v)])).unwrap_err();
        assert_eq!(e.to_string(), msg);
    }
    for calc in ["haversine", "lawOfCosines", "vincentySphere", "cartesian"] {
        let ctx = SpatialContextFactory::make_spatial_context(&args(&[("distCalculator", calc)]))
            .unwrap();
        assert!(!ctx.dist_calc().to_string().is_empty());
    }
    let geo3d = SpatialContextFactory::make_spatial_context(&args(&[
        (
            "spatialContextFactory",
            "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
        ),
        ("planetModel", "clarke1866"),
        ("distCalculator", "geo3d"),
        ("normWrapLongitude", "true"),
        (
            "shapeFactoryClass",
            "org.apache.lucene.spatial.spatial4j.Geo3dShapeFactory",
        ),
        (
            "binaryCodecClass",
            "org.apache.lucene.spatial.spatial4j.Geo3dBinaryCodec",
        ),
    ]))
    .unwrap();
    assert!(geo3d.is_norm_wrap_longitude());
    assert_eq!(geo3d.norm_x(190.0), -170.0);
    assert_eq!(geo3d.norm_y(95.0), 95.0);
    let e = SpatialContextFactory::make_spatial_context(&args(&[
        (
            "spatialContextFactory",
            "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
        ),
        ("planetModel", "mars"),
    ]))
    .unwrap_err();
    assert_eq!(e.to_string(), "Unknown planet model: mars");
    let e = SpatialContextFactory::make_spatial_context(&args(&[("worldBounds", "POINT(1 2)")]))
        .unwrap_err();
    assert_eq!(e.java_class(), "java.lang.ClassCastException");
    // world bounds validation
    let mut f = SpatialContextFactory::new();
    f.world_bounds = Some([-10.0, 10.0, -5.0, 5.0]);
    assert!(f
        .clone()
        .new_spatial_context()
        .unwrap_err()
        .to_string()
        .starts_with("for geo (lat/lon), bounds must be Rect(minX=-180.0"));
    f.geo = false;
    f.world_bounds = Some([10.0, -10.0, -5.0, 5.0]);
    assert!(f
        .clone()
        .new_spatial_context()
        .unwrap_err()
        .to_string()
        .starts_with("worldBounds minX should be <= maxX"));
    f.world_bounds = Some([-10.0, 10.0, 5.0, -5.0]);
    assert!(f
        .clone()
        .new_spatial_context()
        .unwrap_err()
        .to_string()
        .starts_with("worldBounds minY should be <= maxY"));
    assert!(format!("{f:?}").contains("Spatial4j"));
    assert!(SpatialContext::geo_context().is_geo_singleton());
    assert!(!cart().is_geo_singleton());
}

#[test]
fn context_conveniences() {
    let ctx = geo();
    assert!(ctx.verify_x(181.0).is_err() && ctx.verify_x(180.0).is_ok());
    assert!(ctx.verify_y(-91.0).is_err() && ctx.verify_y(90.0).is_ok());
    let a = ctx.point_xy(0.0, 0.0).unwrap();
    let b = ctx.point_xy(10.0, 10.0).unwrap();
    let r = ctx.rect_from_points(&*a, &*b).unwrap();
    assert_eq!((r.min_x(), r.max_y()), (0.0, 10.0));
    assert!(ctx.calc_distance_pts(&*a, &*b).unwrap() > 14.0);
    assert_eq!(
        ctx.binary_codec()
            .read_shape(&ctx, &mut DataInput::new(&[9]))
            .unwrap_err()
            .to_string(),
        "Unsupported shape byte 9"
    );
    assert!(!ctx.is_norm_wrap_longitude());
    assert_eq!(ctx.norm_x(190.0), 190.0);
}

#[test]
fn errors_print_as_java() {
    let cases: Vec<(Error, &str)> = vec![
        (
            Error::InvalidShape("a".into()),
            "org.locationtech.spatial4j.exception.InvalidShapeException: a",
        ),
        (
            Error::Parse {
                message: "b".into(),
                offset: 3,
            },
            "java.text.ParseException: b",
        ),
        (
            Error::IllegalArgument("c".into()),
            "java.lang.IllegalArgumentException: c",
        ),
        (
            Error::UnsupportedOperation(None),
            "java.lang.UnsupportedOperationException",
        ),
        (
            Error::UnsupportedOperation(Some("d".into())),
            "java.lang.UnsupportedOperationException: d",
        ),
        (Error::Runtime("e".into()), "java.lang.RuntimeException: e"),
        (
            Error::ClassCast("f".into()),
            "java.lang.ClassCastException: f",
        ),
        (Error::Io("g".into()), "java.io.IOException: g"),
        (
            Error::NullPointer("h".into()),
            "java.lang.NullPointerException: h",
        ),
        (
            Error::from(crate::spatial3d::Error::IllegalArgument("i".into())),
            "java.lang.IllegalArgumentException: i",
        ),
    ];
    for (e, s) in cases {
        assert_eq!(e.java_to_string(), s);
    }
    assert_eq!(Error::UnsupportedOperation(None).to_string(), "null");
}

#[test]
fn fixed_formatting_rounds_half_up() {
    assert_eq!(java_format_fixed(0.25, 1), "0.3");
    assert_eq!(java_format_fixed(0.15, 1), "0.2");
    assert_eq!(java_format_fixed(2.5, 0), "3");
    assert_eq!(java_format_fixed(9.96, 1), "10.0");
    assert_eq!(java_format_fixed(-0.04, 1), "-0.0");
    assert_eq!(java_format_fixed(1234.5678, 2), "1234.57");
    assert_eq!(java_format_fixed(1e-10, 2), "0.00");
    assert_eq!(java_format_fixed(f64::NAN, 2), "NaN");
    assert_eq!(java_format_fixed(f64::INFINITY, 2), "Infinity");
    assert_eq!(java_format_fixed(f64::NEG_INFINITY, 2), "-Infinity");
    assert_eq!(java_format_fixed(5.0, 2), "5.00");
}

#[test]
fn relations_algebra() {
    use SpatialRelation::*;
    for r in [Within, Contains, Disjoint, Intersects] {
        assert_eq!(r.transpose().transpose(), r);
        assert_eq!(r.combine(None), r);
        assert_eq!(r.combine(Some(r)), r);
        assert_eq!(r.to_string(), r.name());
    }
    assert_eq!(Disjoint.combine(Some(Contains)), Contains);
    assert_eq!(Contains.combine(Some(Disjoint)), Contains);
    assert_eq!(Within.combine(Some(Disjoint)), Intersects);
    assert_eq!(Disjoint.inverse(), Contains);
    assert_eq!(Contains.inverse(), Disjoint);
    assert_eq!(Within.inverse(), Intersects);
    assert!(!Disjoint.intersects());
}

#[test]
fn points_and_rectangles() {
    let ctx = geo();
    let p = PointImpl::without_context(1.0, 2.0);
    assert!(p.context().is_none());
    assert_eq!(
        p.bounding_box().unwrap_err().java_class(),
        "java.lang.NullPointerException"
    );
    assert_eq!((p.lat(), p.lon()), (2.0, 1.0));
    let q = PointImpl::new(1.0, 2.0, ctx.clone());
    assert!(q.equals(&p) && !q.equals(&*ctx.rect(0.0, 1.0, 0.0, 1.0).unwrap()));
    assert!(q.as_any().is::<PointImpl>());
    assert!(!format!("{q:?}").is_empty());
    // vertical lines either side of the dateline relate by their y ranges
    let a = RectangleImpl::new(-180.0, -180.0, 0.0, 10.0, ctx.clone());
    let b = RectangleImpl::new(180.0, 180.0, 2.0, 5.0, ctx.clone());
    assert_eq!(a.relate(&b).unwrap(), SpatialRelation::Contains);
    assert_eq!(b.relate(&a).unwrap(), SpatialRelation::Within);
    let empty = RectangleImpl::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN, ctx.clone());
    assert!(empty.center_point().is_empty());
    assert_eq!(empty.relate(&q).unwrap(), SpatialRelation::Disjoint);
    assert!(a.context().is_some());
    let c = cart();
    let r = RectangleImpl::new(0.0, 10.0, 0.0, 10.0, c.clone());
    assert_eq!(
        r.relate_point(&PointImpl::new(11.0, 5.0, c.clone())),
        SpatialRelation::Disjoint
    );
    assert_eq!(
        r.relate_x_range(-1.0, 11.0).unwrap(),
        SpatialRelation::Within
    );
    assert_eq!(
        r.relate_y_range(2.0, 3.0).unwrap(),
        SpatialRelation::Contains
    );
    assert_eq!(r.area(None).unwrap(), 100.0);
    let circle = c.circle(5.0, 5.0, 1.0).unwrap();
    assert_eq!(r.relate(&*circle).unwrap(), SpatialRelation::Contains);
}

#[test]
fn circles() {
    let c = cart();
    let a = c.circle(0.0, 0.0, 10.0).unwrap();
    let b = c.circle(1.0, 1.0, 2.0).unwrap();
    assert_eq!(a.relate(&*b).unwrap(), SpatialRelation::Contains);
    assert_eq!(b.relate(&*a).unwrap(), SpatialRelation::Within);
    assert_eq!(a.to_string(), "Circle(Pt(x=0.0,y=0.0), d=10.0\u{b0})");
    assert!(a.equals(&*c.circle(0.0, 0.0, 10.0).unwrap()));
    assert!(!a.equals(&*b) && !a.equals(&*c.point_xy(0.0, 0.0).unwrap()));
    let empty_center: Arc<dyn Point> = Arc::new(PointImpl::new(f64::NAN, f64::NAN, c.clone()));
    let e = CircleImpl::new(empty_center, 3.0, c.clone()).unwrap();
    assert!(e.is_empty() && e.radius().is_nan());
    assert_eq!(e.relate(&*a).unwrap(), SpatialRelation::Disjoint);
    assert!(!e.is_geo() && e.center_point().is_empty());
    assert!(e.context().is_some());
    let g = geo().circle(10.0, 10.0, 180.0).unwrap();
    assert!(g.to_string().contains("km"));
    assert_eq!(super::circle::ulp(f64::MAX), 2f64.powi(971));
    assert!(super::circle::ulp(f64::NAN).is_nan());
    assert_eq!(super::circle::ulp(0.0), f64::from_bits(1));
}

#[test]
fn buffered_lines() {
    let c = cart();
    let p = |x, y| c.point_xy(x, y).unwrap();
    let line = BufferedLine::new(p(0.0, 0.0), p(10.0, 0.0), 1.0, c.clone()).unwrap();
    assert_eq!(line.buf(), 1.0);
    assert_eq!(line.a().x(), 0.0);
    assert_eq!(line.b().x(), 10.0);
    assert!(line
        .line_primary()
        .to_string()
        .starts_with("InfBufLine{buf=1.0"));
    assert_eq!(line.line_perp().slope(), f64::NEG_INFINITY);
    assert_eq!(line.line_primary().intercept(), 0.0);
    assert!(line.contains(&*p(5.0, 0.5)));
    assert_eq!(
        line.relate(&*p(5.0, 3.0)).unwrap(),
        SpatialRelation::Disjoint
    );
    assert_eq!(
        line.relate(&*c.circle(0.0, 0.0, 1.0).unwrap()).unwrap_err(),
        Error::UnsupportedOperation(None)
    );
    assert_eq!(line.area(None).unwrap(), 1.0 * 6.0 * 4.0);
    assert!(line.has_area() && !line.is_empty() && line.context().is_some());
    assert_eq!(line.center().unwrap().x(), 5.0);
    let buffered = line.buffered(1.0, &c).unwrap();
    assert!(buffered.to_string().ends_with("b=2.0)"));
    assert!(line.equals(&BufferedLine::new(p(0.0, 0.0), p(10.0, 0.0), 1.0, c.clone()).unwrap()));
    assert!(!line.equals(&*p(0.0, 0.0)));
    // a vertical line: x above / below the intercept
    let v = InfBufLine::new(f64::INFINITY, 2.0, 0.0, 1.0);
    assert_eq!(v.quadrant(3.0, 0.0), 1);
    assert_eq!(v.quadrant(1.0, 0.0), 2);
    assert_eq!(v.distance_unbuffered(4.0, 7.0), 2.0);
    assert!(v.dist_denom_inv().is_nan() && v.buf() == 1.0);
    let ls = BufferedLineString::new(&[p(0.0, 0.0), p(1.0, 1.0)], 0.5, false, c.clone()).unwrap();
    assert_eq!(ls.segments().len(), 1);
    assert_eq!(ls.buf(), 0.5);
    assert!(ls.equals(
        &BufferedLineString::new(&[p(0.0, 0.0), p(1.0, 1.0)], 0.5, false, c.clone()).unwrap()
    ));
    assert!(!ls.equals(&line));
    assert!(ls.context().is_some() && ls.as_any().is::<BufferedLineString>());
    let empty = BufferedLineString::new(&[], 0.5, false, c.clone()).unwrap();
    assert!(empty.is_empty() && empty.points().is_empty());
    assert_eq!(empty.to_string(), "BufferedLineString(buf=0.5 pts=)");
    assert!(!format!("{line:?}").is_empty());
    let skew = BufferedLine::expand_buf_for_longitude_skew(&*p(0.0, 60.0), &*p(1.0, 10.0), 1.0);
    assert!(skew > 1.0);
}

#[test]
fn collections() {
    let c = cart();
    let shapes: Vec<Arc<dyn Shape>> = (0..40)
        .map(|i| c.point_xy(i as f64, 0.0).unwrap() as Arc<dyn Shape>)
        .collect();
    let coll = ShapeCollection::new(shapes.clone(), c.clone()).unwrap();
    assert!(coll.to_string().ends_with(" ...40)"));
    assert_eq!(coll.size(), 40);
    assert_eq!(coll.get(3).center().unwrap().x(), 3.0);
    assert!(coll.equals(&ShapeCollection::new(shapes, c.clone()).unwrap()));
    assert!(!coll.equals(&*c.point_xy(0.0, 0.0).unwrap()));
    assert!(coll.context().is_some());
    let empty = ShapeCollection::new(Vec::new(), c.clone()).unwrap();
    assert_eq!(
        empty.relate(&*c.point_xy(0.0, 0.0).unwrap()).unwrap(),
        SpatialRelation::Disjoint
    );
    let r = coll.relate(&*c.point_xy(5.0, 0.0).unwrap()).unwrap();
    assert_eq!(r, SpatialRelation::Intersects);
}

#[test]
fn bbox_calculator_orders_like_double_compare() {
    use super::bbox_calculator::BBoxCalculator;
    let mut calc = BBoxCalculator::new(geo());
    calc.expand_range(-0.0, 0.0, 0.0, 1.0);
    calc.expand_range(10.0, 20.0, -1.0, 0.5);
    calc.expand_range(170.0, -170.0, 0.0, 0.0);
    assert_eq!((calc.min_y(), calc.max_y()), (-1.0, 1.0));
    // the biggest gap is -170..-0, so the box runs from -0 east to -170
    assert_eq!((calc.min_x(), calc.max_x()), (-0.0, -170.0));
    assert!(calc.min_x().is_sign_negative());
    let mut world = BBoxCalculator::new(geo());
    world.expand_x_range(-170.0, 170.0);
    world.expand_x_range(160.0, -160.0);
    assert!(world.does_x_world_wrap());
    world.expand_range(0.0, 1.0, 0.0, 1.0);
    assert_eq!(world.boundary().unwrap().width(), 360.0);
    let mut nan = BBoxCalculator::new(geo());
    nan.expand_x_range(f64::NAN, f64::NAN);
    nan.expand_x_range(1.0, 2.0);
    // NaN sorts last and spoils every gap, as in Java: no box is chosen.
    assert_eq!(nan.max_x(), f64::NEG_INFINITY);
}

#[test]
fn distance_calculators() {
    let c = cart();
    let calc = CartesianDistCalc::new(false);
    let p = c.point_xy(0.0, 5.0).unwrap();
    assert_eq!(calc.distance_to_line_segment(&*p, -1.0, 0.0, 1.0, 0.0), 5.0);
    assert_eq!(
        calc.distance_to_line_segment(&*p, 3.0, 0.0, 3.0, 0.0),
        (9.0f64 + 25.0).sqrt()
    );
    assert_eq!(calc.distance_to_line_segment(&*p, 4.0, 5.0, 8.0, 5.0), 4.0);
    assert_eq!(
        calc.distance_to_line_segment(&*p, -8.0, 5.0, -4.0, 5.0),
        4.0
    );
    assert!(calc.equals(&CartesianDistCalc::new(false)));
    assert!(!calc.equals(&CartesianDistCalc::new(true)));
    assert!(!calc.equals(&GeodesicSphereDistCalc::Haversine));
    assert!(GeodesicSphereDistCalc::Vincenty.equals(&GeodesicSphereDistCalc::Vincenty));
    let same = calc.point_on_bearing(&p, 0.0, 30.0, &c).unwrap();
    assert!(Arc::ptr_eq(&same, &p));
    let g = geo();
    let gp = g.point_xy(1.0, 2.0).unwrap();
    let same = GeodesicSphereDistCalc::Haversine
        .point_on_bearing(&gp, 0.0, 30.0, &g)
        .unwrap();
    assert!(Arc::ptr_eq(&same, &gp));
    let r = g.rect(0.0, 10.0, 0.0, 10.0).unwrap();
    assert!(GeodesicSphereDistCalc::Haversine.area_rect(&*r).unwrap() > 0.0);
    assert_eq!(
        calc.area_rect(&*c.rect(0.0, 2.0, 0.0, 3.0).unwrap())
            .unwrap(),
        6.0
    );
    assert_eq!(DistanceUtils::radians2_dist(2.0, 3.0), 6.0);
    assert_eq!(DistanceUtils::dist2_radians(6.0, 3.0), 2.0);
    assert_eq!(DistanceUtils::norm_lon_deg(540.0), 180.0);
    assert_eq!(DistanceUtils::norm_lon_deg(-540.0), -180.0);
    assert_eq!(DistanceUtils::norm_lat_deg(100.0), 80.0);
    assert_eq!(DistanceUtils::norm_lat_deg(300.0), -60.0);
    assert_eq!(
        DistanceUtils::calc_box_by_dist_from_pt_lat_horiz_axis_deg(0.0, 0.0, 0.0),
        0.0
    );
    let pole = DistanceUtils::point_on_bearing_rad(1.5, 3.0, 0.2, 0.0);
    assert!(pole.1 <= DistanceUtils::DEG_90_AS_RADS);
    let south = DistanceUtils::point_on_bearing_rad(-1.5, -3.0, 0.2, std::f64::consts::PI);
    assert!(south.1 >= -DistanceUtils::DEG_90_AS_RADS);
    let wrap = DistanceUtils::point_on_bearing_rad(0.0, 3.1, 0.2, std::f64::consts::FRAC_PI_2);
    assert!(wrap.0 < 0.0);
    let wrap = DistanceUtils::point_on_bearing_rad(0.0, -3.1, 0.2, -std::f64::consts::FRAC_PI_2);
    assert!(wrap.0 > 0.0);
    assert_eq!(
        DistanceUtils::calc_box_by_dist_from_pt_lat_horiz_axis_deg(0.0, 0.0, 179.0),
        90.0
    );
    assert!((DistanceUtils::KM_TO_DEG * DistanceUtils::DEG_TO_KM - 1.0).abs() < 1e-15);
    assert!((DistanceUtils::MILES_TO_KM * DistanceUtils::KM_TO_MILES - 1.0).abs() < 1e-15);
    assert!((DistanceUtils::EARTH_MEAN_RADIUS_MI - 3958.76).abs() < 0.01);
    assert!((DistanceUtils::EARTH_EQUATORIAL_RADIUS_MI - 3963.19).abs() < 0.01);
}

#[test]
fn binary_codec_edges() {
    let c = cart();
    let codec = c.binary_codec();
    assert_eq!(
        codec
            .read_shape(&c, &mut DataInput::new(&[1, 0]))
            .unwrap_err()
            .to_string(),
        "java.io.EOFException"
    );
    // a collection whose members share a declared type byte
    let mut bytes = vec![4u8, 1];
    bytes.extend_from_slice(&2i32.to_be_bytes());
    for v in [1.0, 2.0, 3.0, 4.0] {
        write_double(&mut bytes, v);
    }
    let s = codec.read_shape(&c, &mut DataInput::new(&bytes)).unwrap();
    assert_eq!(
        s.to_string(),
        "ShapeCollection(Pt(x=1.0,y=2.0), Pt(x=3.0,y=4.0))"
    );
    let mut bad = vec![4u8, 9];
    bad.extend_from_slice(&1i32.to_be_bytes());
    assert_eq!(
        codec
            .read_shape(&c, &mut DataInput::new(&bad))
            .unwrap_err()
            .to_string(),
        "Unsupported shape byte 9"
    );
    let mut nan = Vec::new();
    write_double(&mut nan, f64::from_bits(0xfff8_0000_0000_0001));
    assert_eq!(nan, 0x7ff8_0000_0000_0000u64.to_be_bytes());
    let mut input = DataInput::new(&[0, 0, 0, 7, 1]);
    assert_eq!(input.read_int().unwrap(), 7);
    assert_eq!(input.position(), 4);
    assert_eq!(input.read_byte().unwrap(), 1);
    let line = c
        .line_string(&[c.point_xy(0.0, 0.0).unwrap()], 1.0)
        .unwrap();
    assert_eq!(
        codec
            .write_shape(&mut Vec::new(), &*line)
            .unwrap_err()
            .to_string(),
        "Unsupported shape class org.locationtech.spatial4j.shape.impl.BufferedLineString"
    );
    let seg = BufferedLine::new(
        c.point_xy(0.0, 0.0).unwrap(),
        c.point_xy(1.0, 0.0).unwrap(),
        1.0,
        c.clone(),
    )
    .unwrap();
    assert!(codec
        .write_shape(&mut Vec::new(), &seg)
        .unwrap_err()
        .to_string()
        .ends_with("BufferedLine"));
    assert_eq!(
        super::binary_codec::java_class_name(&*c.point_xy(0.0, 0.0).unwrap()),
        "org.locationtech.spatial4j.shape.Shape"
    );
}

#[test]
fn geohash_edges() {
    assert!(geohash::decode_boundary_values("a").is_err());
    assert_eq!(
        geohash::lookup_hash_len_for_width_height(0.0, 0.0),
        geohash::MAX_PRECISION
    );
}

#[test]
fn wkt_lexing() {
    assert!(java_is_whitespace('\u{1F}') && java_is_whitespace('\u{2028}'));
    assert!(!java_is_whitespace('\u{A0}') && !java_is_whitespace('x'));
    assert_eq!(
        java_parse_double("").unwrap_err(),
        "java.lang.NumberFormatException: empty String"
    );
    assert_eq!(
        java_parse_double("1x").unwrap_err(),
        "java.lang.NumberFormatException: For input string: \"1x\""
    );
    let ctx = geo();
    let reader = ctx.wkt_reader();
    assert!(reader.parse_if_supported("  ").unwrap().is_none());
    let long = format!("FOO({})", "1 ".repeat(100));
    let e = reader.parse(&long).unwrap_err();
    assert!(e.to_string().ends_with("...]"), "{e}");
    let mut state = super::wkt::State::new("A(B(1,2)),C");
    assert_eq!(state.next_sub_shape_string().unwrap(), "A(B(1,2))");
    let mut bad = super::wkt::State::new("A(B");
    assert_eq!(
        bad.next_sub_shape_string().unwrap_err().to_string(),
        "Unbalanced parenthesis"
    );
    let mut word = super::wkt::State::new("(");
    assert_eq!(word.next_word().unwrap_err().to_string(), "Word expected");
    let geo3d = SpatialContextFactory::geo3d()
        .new_spatial_context()
        .unwrap();
    let e = geo3d
        .read_shape_from_wkt("GEOMETRYCOLLECTION(LINESTRING EMPTY)")
        .unwrap_err();
    assert_eq!(e.java_class(), "java.text.ParseException");
}

#[test]
fn factory_builders() {
    let ctx = geo();
    let f = ctx.shape_factory();
    let mut ls = f.line_string_builder(&ctx);
    ls.point_xyz(1.0, 2.0, 3.0).unwrap();
    ls.point_lat_lon(4.0, 3.0).unwrap();
    ls.buffer(0.5);
    let built = ls.build().unwrap();
    assert!(built
        .to_string()
        .starts_with("BufferedLineString(buf=0.5 pts=1.0 2.0, 3.0 4.0"));
    let mut mp = f.multi_point_builder(&ctx);
    mp.point_xyz(1.0, 2.0, 9.0).unwrap();
    assert_eq!(
        mp.build().unwrap().to_string(),
        "ShapeCollection(Pt(x=1.0,y=2.0))"
    );
    assert_eq!(f.point_lat_lon(&ctx, 1.0, 2.0).unwrap().x(), 2.0);
    assert_eq!(f.point_xyz(&ctx, 1.0, 2.0, 3.0).unwrap().y(), 2.0);
    assert_eq!(f.norm_z(7.0), 7.0);
    assert!(f.as_s2().is_none());
    let mut msb = f.multi_shape_builder(&ctx);
    msb.add(ctx.point_xy(1.0, 1.0).unwrap()).unwrap();
    assert!(msb.build().is_ok());
    let mpb = f.multi_polygon_builder(&ctx);
    assert_eq!(
        mpb.polygon().err().unwrap().java_class(),
        "java.lang.UnsupportedOperationException"
    );
    assert!(mpb.build().is_ok());
    let c = cart();
    let e = c.rect(5.0, 1.0, 0.0, 1.0).unwrap_err();
    assert_eq!(e.to_string(), "maxX must be >= minX: 5.0 to 1.0");
    let bounded = SpatialContextFactory::make_spatial_context(&args(&[
        ("geo", "false"),
        ("worldBounds", "ENVELOPE(-1, 1, 1, -1)"),
    ]))
    .unwrap();
    assert!(bounded
        .rect(-2.0, 0.0, 0.0, 0.5)
        .unwrap_err()
        .to_string()
        .starts_with("X values"));
    assert!(c.circle(0.0, 0.0, -1.0).is_err());
    assert_eq!(ctx.circle(0.0, 0.0, 200.0).unwrap().radius(), 180.0);
    let mut ml = f.multi_line_string_builder(&ctx);
    let mut one = ml.line_string();
    one.point_xy(0.0, 0.0).unwrap();
    ml.add(one).unwrap();
    assert!(ml
        .build()
        .unwrap()
        .to_string()
        .starts_with("ShapeCollection(BufferedLineString"));
}

/// Nested collections in a binary shape and nested shapes in WKT:
/// `MAX_NESTING` deep reads, one deeper is the `StackOverflowError` Java
/// would hit only at its stack's end (a Rust stack overflow aborts).
#[test]
fn nesting_stops_at_the_limit() {
    let c = cart();
    let codec = c.binary_codec();
    let nested = |depth: u32| {
        let mut out = Vec::new();
        for _ in 0..depth {
            out.extend_from_slice(&[4, 0]); // a collection of typed members
            out.extend_from_slice(&1i32.to_be_bytes());
        }
        out.push(1);
        write_double(&mut out, 1.0);
        write_double(&mut out, 2.0);
        out
    };
    let limit = binary_codec::MAX_NESTING;
    assert!(codec
        .read_shape(&c, &mut DataInput::new(&nested(limit)))
        .is_ok());
    let e = codec
        .read_shape(&c, &mut DataInput::new(&nested(limit + 1)))
        .unwrap_err();
    assert!(
        e.to_string().starts_with("java.lang.StackOverflowError"),
        "{e}"
    );
    let mut input = DataInput::new(&[1, 2, 3]);
    input.advance(usize::MAX);
    assert!(input.remaining().is_empty());

    let wkt = |depth: u32| {
        let d = depth as usize;
        format!(
            "{}POINT(1 2){}",
            "GEOMETRYCOLLECTION(".repeat(d),
            ")".repeat(d)
        )
    };
    let limit = wkt::MAX_NESTING;
    assert!(c.read_shape_from_wkt(&wkt(limit)).is_ok());
    let e = c.read_shape_from_wkt(&wkt(limit + 1)).unwrap_err();
    assert_eq!(e.java_class(), "java.text.ParseException");
    assert!(e.to_string().contains("StackOverflowError"), "{e}");
}
