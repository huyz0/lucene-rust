//! Unit tests for the Geo3D bridge: the paths the differential fixtures
//! (`tests/spatial4j_fixtures.rs`, `tests/spatial_prefix_tree_fixtures.rs`)
//! do not reach.

use std::sync::Arc;

use super::*;
use crate::spatial4j::{BufferedLine, SpatialContextFactory};

fn ctx() -> Arc<SpatialContext> {
    SpatialContextFactory::geo3d()
        .new_spatial_context()
        .unwrap()
}

#[test]
fn shapes_of_each_kind() {
    let ctx = ctx();
    let p = ctx.point_xy(10.0, 20.0).unwrap();
    assert!((p.x() - 10.0).abs() < 1e-12 && (p.y() - 20.0).abs() < 1e-12);
    assert!(!p.has_area() && !p.is_empty());
    assert_eq!(p.to_string(), "Geo3D:GeoDegeneratePoint");
    assert_eq!(format!("{p:?}"), "Geo3D:GeoDegeneratePoint");
    assert_eq!(
        geo3d_class_name(&*p),
        Some("org.apache.lucene.spatial.spatial4j.Geo3dPointShape")
    );
    assert!(Arc::ptr_eq(p.context().unwrap(), &ctx));
    assert_eq!(p.area(None).unwrap_err(), Error::UnsupportedOperation(None));
    let pb = p.bounding_box().unwrap();
    assert!((pb.min_x() - 10.0).abs() < 1e-12);
    assert!((p.center().unwrap().x() - 10.0).abs() < 1e-12);
    let buffered = p.buffered(1.0, &ctx).unwrap();
    assert!((buffered.as_circle().unwrap().radius() - 1.0).abs() < 1e-12);

    let r = ctx.rect(-10.0, 10.0, -5.0, 5.0).unwrap();
    assert_eq!(
        (r.min_x(), r.max_x(), r.min_y(), r.max_y()),
        (-10.0, 10.0, -5.0, 5.0)
    );
    assert_eq!((r.width(), r.height()), (20.0, 10.0));
    assert!(
        r.crosses_date_line(),
        "Lucene's test is maxX > 0 && minX < 0"
    );
    let wrap = ctx.rect(170.0, -170.0, -5.0, 5.0).unwrap();
    assert_eq!(wrap.width(), 20.0);
    assert_eq!(
        r.relate_y_range(-1.0, 1.0).unwrap(),
        SpatialRelation::Intersects
    );
    assert_eq!(
        r.relate_x_range(-1.0, 1.0).unwrap(),
        SpatialRelation::Intersects
    );
    assert!(r.center().unwrap().x().abs() < 1e-12);
    assert!(
        Arc::ptr_eq(&r.center().unwrap(), &r.center().unwrap()),
        "cached"
    );
    let rb = r.buffered(1.0, &ctx).unwrap();
    assert!(rb.as_rectangle().unwrap().min_x() < -10.0);
    assert!(r.as_point().is_none() && r.as_circle().is_none());
    let rg = r.as_any().downcast_ref::<Geo3dShape>().unwrap();
    assert!(Point::x(rg).is_nan() && Point::y(rg).is_nan() && Circle::radius(rg).is_nan());
    let pg = p.as_any().downcast_ref::<Geo3dShape>().unwrap();
    assert!(Rectangle::min_x(pg).is_nan());

    let c = ctx.circle(0.0, 0.0, 2.0).unwrap();
    assert!((c.radius() - 2.0).abs() < 1e-12);
    assert!(c.center().unwrap().y().abs() < 1e-12);
    assert!(c.has_area());
    assert_eq!(
        c.buffered(1.0, &ctx).unwrap_err(),
        Error::UnsupportedOperation(None)
    );
    assert_eq!(c.relate(&*p).unwrap(), SpatialRelation::Disjoint);
    assert_eq!(
        c.relate(&*ctx.point_xy(0.5, 0.5).unwrap()).unwrap(),
        SpatialRelation::Contains
    );

    let poly = ctx
        .read_shape_from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
        .unwrap();
    let pc = poly.center().unwrap();
    assert!(pc.x() > 0.0 && pc.x() < 10.0);
    assert!(poly.equals(
        &*ctx
            .read_shape_from_wkt("POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))")
            .unwrap()
    ));
    assert!(!poly.equals(&*p));
    assert!(!poly.equals(
        &*crate::spatial4j::SpatialContext::geo_context()
            .point_xy(0.0, 0.0)
            .unwrap()
    ));
    let copy = poly.as_any().downcast_ref::<Geo3dShape>().unwrap().clone();
    assert!(copy.equals(&*poly));
    // relating to a shape Geo3D cannot relate
    let cart = SpatialContextFactory::make_spatial_context(
        &[("geo".to_string(), "false".to_string())]
            .into_iter()
            .collect(),
    )
    .unwrap();
    let line = BufferedLine::new(
        cart.point_xy(0.0, 0.0).unwrap(),
        cart.point_xy(1.0, 1.0).unwrap(),
        1.0,
        cart.clone(),
    )
    .unwrap();
    assert_eq!(
        poly.relate(&line).unwrap_err().to_string(),
        "Unimplemented shape relationship determination: class org.locationtech.spatial4j.shape.impl.BufferedLine"
    );
}

#[test]
fn relates_to_plain_spatial4j_shapes() {
    let ctx = ctx();
    let geo = crate::spatial4j::SpatialContext::geo_context();
    let c = ctx.circle(0.0, 0.0, 10.0).unwrap();
    let inside = geo.rect(-1.0, 1.0, -1.0, 1.0).unwrap();
    // Lucene reads `GeoArea.getRelationship` the wrong way round for a
    // rectangle that is not a Geo3D shape: the circle containing the box
    // comes out WITHIN, and the world containing the circle CONTAINS.
    assert_eq!(c.relate(&*inside).unwrap(), SpatialRelation::Within);
    let world = ctx.world_bounds();
    assert_eq!(c.relate(&world).unwrap(), SpatialRelation::Contains);
    // the same box as a Geo3D rectangle relates the right way round
    let r3 = ctx.rect(-1.0, 1.0, -1.0, 1.0).unwrap();
    assert_eq!(c.relate(&*r3).unwrap(), SpatialRelation::Contains);
    let far = geo.point_xy(100.0, 0.0).unwrap();
    assert_eq!(c.relate(&*far).unwrap(), SpatialRelation::Disjoint);
    let near = geo.point_xy(1.0, 1.0).unwrap();
    assert_eq!(c.relate(&*near).unwrap(), SpatialRelation::Contains);
    let g = c.as_any().downcast_ref::<Geo3dShape>().unwrap();
    assert!(g.geo_shape().planet_model().is_sphere());
    assert!(g.geo_point_shape().is_none());
}

#[test]
fn distance_calculator() {
    let ctx = ctx();
    let calc = ctx.dist_calc();
    assert_eq!(calc.to_string(), "Geo3dDistanceCalculator");
    assert!(calc.equals(&**calc));
    assert!(!calc.equals(&crate::spatial4j::CartesianDistCalc::new(false)));
    let a = ctx.point_xy(0.0, 0.0).unwrap();
    let b = ctx.point_xy(0.0, 1.0).unwrap();
    let d = calc.distance(&*a, &*b).unwrap();
    assert!((d - 1.0).abs() < 1e-9);
    let plain = crate::spatial4j::PointImpl::without_context(0.0, 1.0);
    assert!((calc.distance(&plain, &*b).unwrap()).abs() < 1e-9);
    assert!(calc.within(&*a, 0.0, 1.0, 0.5).unwrap());
    let moved = calc.point_on_bearing(&a, 1.0, 90.0, &ctx).unwrap();
    assert!((moved.x() - 1.0).abs() < 1e-9);
    let plain: Arc<dyn Point> = Arc::new(crate::spatial4j::PointImpl::without_context(0.0, 1.0));
    assert_eq!(
        calc.point_on_bearing(&plain, 1.0, 0.0, &ctx)
            .unwrap_err()
            .java_class(),
        "java.lang.ClassCastException"
    );
    assert!(
        calc.calc_box_by_dist_from_pt(&a, 1.0, &ctx)
            .unwrap()
            .max_y()
            > 0.9
    );
    assert!(calc
        .calc_box_by_dist_from_pt_y_horiz_axis_deg(&*a, 1.0, &ctx)
        .is_err());
    let r = ctx.rect(0.0, 1.0, 0.0, 1.0).unwrap();
    assert!(calc.area_rect(&*r).is_err());
    let c = ctx.circle(0.0, 0.0, 1.0).unwrap();
    assert!(calc.area_circle(&*c).is_err());
}

#[test]
fn binary_codec_edges() {
    let ctx = ctx();
    let codec = ctx.binary_codec();
    let mut bytes = vec![1u8, 0];
    GeoPoint::new(1.0, 0.0, 0.0).write(&mut bytes);
    let e = codec
        .read_shape(&ctx, &mut DataInput::new(&bytes))
        .unwrap_err();
    assert_eq!(
        e.to_string(),
        "trying to read a not supported shape: class org.apache.lucene.spatial3d.geom.GeoPoint"
    );
    let plain = crate::spatial4j::PointImpl::without_context(0.0, 1.0);
    assert!(codec
        .write_shape(&mut Vec::new(), &plain)
        .unwrap_err()
        .to_string()
        .starts_with("trying to write a not supported shape"));
}

#[test]
fn factory_and_builders() {
    let ctx = ctx();
    let f = ctx.shape_factory();
    assert!(!f.is_norm_wrap_longitude());
    assert_eq!(f.norm_x(190.0), 190.0);
    let p = f.point_xyz(&ctx, 1.0, 0.0, 0.0).unwrap();
    assert!(p.x().abs() < 1e-12 && p.y().abs() < 1e-12);
    assert!(f.point_xy_unchecked(&ctx, 1.0, 2.0).is_ok());
    assert_eq!(
        f.multi_shape(&ctx, Vec::new()).unwrap_err(),
        Error::UnsupportedOperation(None)
    );
    let line = f
        .line_string(
            &ctx,
            &[
                ctx.point_xy(0.0, 0.0).unwrap(),
                ctx.point_xy(1.0, 1.0).unwrap(),
            ],
            0.01,
        )
        .unwrap();
    assert!(line.has_area());
    let mut mp = f.multi_point_builder(&ctx);
    mp.point_xyz(1.0, 0.0, 0.0).unwrap();
    mp.point_xyz(1.0, 0.0, 0.0).unwrap();
    mp.point_xy(5.0, 5.0).unwrap();
    assert!(mp.build().is_ok());
    let mut ls = f.line_string_builder(&ctx);
    ls.point_xyz(1.0, 0.0, 0.0).unwrap();
    ls.point_xyz(0.0, 1.0, 0.0).unwrap();
    assert!(ls.build().is_ok());
    let mut poly = f.polygon_builder(&ctx).unwrap();
    for (x, y) in [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)] {
        poly.point_xy(x, y).unwrap();
    }
    poly.point_xyz(1.0, 0.0, 0.0).unwrap();
    {
        let mut hole = poly.hole();
        hole.point_xyz(0.9, 0.1, 0.1).unwrap();
        for (x, y) in [(2.0, 2.0), (4.0, 2.0), (4.0, 4.0)] {
            hole.point_xy(x, y).unwrap();
        }
        hole.end_hole().unwrap();
    }
    let _ = poly.build();
    let mut msb = f.multi_shape_builder(&ctx);
    let plain = Arc::new(crate::spatial4j::PointImpl::without_context(0.0, 1.0));
    assert_eq!(
        msb.add(plain).unwrap_err().java_class(),
        "java.lang.ClassCastException"
    );
    msb.add(ctx.point_xy(1.0, 1.0).unwrap()).unwrap();
    assert!(msb.build().is_ok());
    let mut factory = Geo3dShapeFactory::new(PlanetModel::wgs84(), false);
    factory.set_circle_accuracy(1e-3);
    assert!(!factory.planet_model().is_sphere());
    let wgs = SpatialContextFactory {
        planet_model: Some(PlanetModel::wgs84()),
        ..SpatialContextFactory::geo3d()
    }
    .new_spatial_context()
    .unwrap();
    assert!(
        wgs.circle(0.0, 0.0, 1.0).is_ok(),
        "an exact circle off the sphere"
    );
    let s2 = f.as_s2().unwrap();
    let cell = s2
        .s2_cell_shape(&ctx, crate::s2::S2CellId::from_face_pos_level(0, 0, 3))
        .unwrap();
    assert!(cell.has_area());
}
