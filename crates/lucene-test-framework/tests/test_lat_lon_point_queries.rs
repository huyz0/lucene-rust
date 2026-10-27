//! Lucene's `BaseGeoPointTestCase` subclasses `TestLatLonPointQueries`
//! (BKD points) and `TestLatLonDocValuesQueries` (the slow doc-values
//! queries), over the random tests the base ports; and `GeoTestUtil`'s
//! generators.

use std::sync::Arc;

use lucene_index::document::{Document, LatLonDocValuesField, LatLonPoint};
use lucene_search::document::geo::{lat_lon_doc_values_field, lat_lon_point, QueryRelation};
use lucene_search::document::DocumentQuery;
use lucene_search::extended_query::DocumentClause;
use lucene_search::query::Clause;
use lucene_test_framework::geo::base_geo_point::{rect_contains_point, BaseGeoPoint, GeoPointCase};
use lucene_test_framework::geo::geo_test_util::{
    java_next_after_f32, next_box, next_box_not_crossing_dateline, next_circle,
    next_double_internal, Rect,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::geo::{GeoEncodingUtils, LatLonGeometry, Polygon};
use lucene_util::java_random::JavaRandom;

fn clause(key: String, q: Box<dyn DocumentQuery>) -> Clause {
    DocumentClause::new(key, Arc::from(q)).into()
}

fn quantize_lat(lat: f64) -> f64 {
    GeoEncodingUtils::decode_latitude(GeoEncodingUtils::encode_latitude(lat).expect("a latitude"))
}

fn quantize_lon(lon: f64) -> f64 {
    GeoEncodingUtils::decode_longitude(
        GeoEncodingUtils::encode_longitude(lon).expect("a longitude"),
    )
}

fn rect(
    field: &str,
    r: &Rect,
    q: lucene_search::Result<Box<dyn DocumentQuery>>,
) -> Result<Clause, String> {
    Ok(clause(
        format!("box {field} {r:?}"),
        q.map_err(|e| e.to_string())?,
    ))
}

fn keyed(key: String, q: lucene_search::Result<Box<dyn DocumentQuery>>) -> Result<Clause, String> {
    Ok(clause(key, q.map_err(|e| e.to_string())?))
}

/// `TestLatLonPointQueries`.
struct PointCase;

impl GeoPointCase for PointCase {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        lat: f64,
        lon: f64,
    ) -> Result<(), String> {
        doc.add(LatLonPoint::new(field, lat, lon).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn new_rect_query(&self, field: &str, r: &Rect) -> Result<Clause, String> {
        let q = lat_lon_point::new_box_query(field, r.min_lat, r.max_lat, r.min_lon, r.max_lon);
        rect(field, r, q)
    }

    fn new_distance_query(
        &self,
        field: &str,
        lat: f64,
        lon: f64,
        radius: f64,
    ) -> Result<Clause, String> {
        let q = lat_lon_point::new_distance_query(field, lat, lon, radius);
        keyed(format!("circle {field} {lat} {lon} {radius}"), q)
    }

    fn new_polygon_query(&self, field: &str, polygons: &[Polygon]) -> Result<Clause, String> {
        let q = lat_lon_point::new_polygon_query(field, polygons);
        keyed(format!("polygon {field} {polygons:?}"), q)
    }

    fn new_geometry_query(
        &self,
        field: &str,
        geometries: &[LatLonGeometry],
    ) -> Result<Clause, String> {
        let q = lat_lon_point::new_geometry_query(field, QueryRelation::Intersects, geometries);
        keyed(format!("geometry {field} {geometries:?}"), q)
    }

    fn quantize_lat(&self, lat: f64) -> f64 {
        quantize_lat(lat)
    }

    fn quantize_lon(&self, lon: f64) -> f64 {
        quantize_lon(lon)
    }
}

/// `TestLatLonDocValuesQueries`.
struct DocValuesCase;

impl GeoPointCase for DocValuesCase {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        lat: f64,
        lon: f64,
    ) -> Result<(), String> {
        doc.add(LatLonDocValuesField::new(field, lat, lon).map_err(|e| e.to_string())?);
        Ok(())
    }

    fn new_rect_query(&self, field: &str, r: &Rect) -> Result<Clause, String> {
        let q = lat_lon_doc_values_field::new_slow_box_query(
            field, r.min_lat, r.max_lat, r.min_lon, r.max_lon,
        );
        rect(&format!("slow {field}"), r, q)
    }

    fn new_distance_query(
        &self,
        field: &str,
        lat: f64,
        lon: f64,
        radius: f64,
    ) -> Result<Clause, String> {
        let q = lat_lon_doc_values_field::new_slow_distance_query(field, lat, lon, radius);
        keyed(format!("slow circle {field} {lat} {lon} {radius}"), q)
    }

    fn new_polygon_query(&self, field: &str, polygons: &[Polygon]) -> Result<Clause, String> {
        let q = lat_lon_doc_values_field::new_slow_polygon_query(field, polygons);
        keyed(format!("slow polygon {field} {polygons:?}"), q)
    }

    fn new_geometry_query(
        &self,
        field: &str,
        geometries: &[LatLonGeometry],
    ) -> Result<Clause, String> {
        let q = lat_lon_doc_values_field::new_slow_geometry_query(
            field,
            QueryRelation::Intersects,
            geometries,
        );
        keyed(format!("slow geometry {field} {geometries:?}"), q)
    }

    fn quantize_lat(&self, lat: f64) -> f64 {
        quantize_lat(lat)
    }

    fn quantize_lon(&self, lon: f64) -> f64 {
        quantize_lon(lon)
    }
}

/// Every test method of the base, each on its own stream; the nightly
/// ones only under `tests.nightly`.
fn run_all<C: GeoPointCase>(name: &str, make: impl Fn() -> C) {
    type Method<C> = fn(&mut BaseGeoPoint<'_, C>);
    let methods: [(&str, Method<C>, bool); 16] = [
        ("random_tiny", |b| b.random_tiny(), false),
        ("random_medium", |b| b.random_medium(), false),
        ("random_big", |b| b.random_big(), true),
        (
            "same_point_many_times",
            |b| b.same_point_many_times(),
            false,
        ),
        ("low_cardinality", |b| b.low_cardinality(), false),
        ("all_lat_equal", |b| b.all_lat_equal(), false),
        ("all_lon_equal", |b| b.all_lon_equal(), false),
        ("multi_valued", |b| b.multi_valued(), false),
        (
            "rect_boundaries_are_inclusive",
            |b| b.rect_boundaries_are_inclusive(),
            false,
        ),
        ("random_distance", |b| b.random_distance(), false),
        ("random_distance_huge", |b| b.random_distance_huge(), true),
        ("basics", |b| b.basics(), false),
        ("index_values", |b| b.index_values(), false),
        ("invalid_queries", |b| b.invalid_queries(), false),
        ("small_set", |b| b.small_set(), false),
        ("equals", |b| b.equals(), false),
    ];
    for (method, run, nightly) in methods {
        let mut tc = LuceneTestCase::new(&format!("{name}::{method}"));
        if nightly && !tc.nightly() {
            continue;
        }
        let multiplier = tc.multiplier();
        run(&mut BaseGeoPoint::new(make(), tc.random(), multiplier));
    }
    let mut tc = LuceneTestCase::new(&format!("{name}::narrow_polygon_close_to_north_pole"));
    let multiplier = tc.multiplier();
    BaseGeoPoint::new(make(), tc.random(), multiplier).narrow_polygon_close_to_north_pole();
}

#[test]
fn lat_lon_point_queries() {
    run_all("TestLatLonPointQueries", || PointCase);
}

#[test]
fn lat_lon_doc_values_queries() {
    run_all("TestLatLonDocValuesQueries", || DocValuesCase);
}

/// The nightly methods, at least once per run (the doc-values case is
/// the cheaper to search, and both share the base's code).
#[test]
fn nightly_methods_run() {
    let mut r = JavaRandom::new(3);
    BaseGeoPoint::new(DocValuesCase, &mut r, 1).do_random_distance_test(2000, 10);
}

/// [`PointCase`] with one kind of query answering the wrong question:
/// boxes the whole world, circles a metre at the origin, polygons and
/// geometries a tiny square there.
struct Wrong(&'static str);

impl Wrong {
    fn tiny() -> Polygon {
        Polygon::new(
            &[0.0, 0.0, 0.1, 0.1, 0.0],
            &[0.0, 0.1, 0.1, 0.0, 0.0],
            vec![],
        )
        .unwrap()
    }
}

impl GeoPointCase for Wrong {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        lat: f64,
        lon: f64,
    ) -> Result<(), String> {
        PointCase.add_point_to_doc(field, doc, lat, lon)
    }
    fn new_rect_query(&self, field: &str, r: &Rect) -> Result<Clause, String> {
        let world = Rect {
            min_lat: -90.0,
            max_lat: 90.0,
            min_lon: -180.0,
            max_lon: 180.0,
        };
        PointCase.new_rect_query(field, if self.0 == "box" { &world } else { r })
    }
    fn new_distance_query(
        &self,
        field: &str,
        lat: f64,
        lon: f64,
        radius: f64,
    ) -> Result<Clause, String> {
        if self.0 == "circle" {
            PointCase.new_distance_query(field, 0.0, 0.0, 1.0)
        } else {
            PointCase.new_distance_query(field, lat, lon, radius)
        }
    }
    fn new_polygon_query(&self, field: &str, polygons: &[Polygon]) -> Result<Clause, String> {
        if self.0 == "polygon" {
            PointCase.new_polygon_query(field, &[Self::tiny()])
        } else {
            PointCase.new_polygon_query(field, polygons)
        }
    }
    fn new_geometry_query(
        &self,
        field: &str,
        geometries: &[LatLonGeometry],
    ) -> Result<Clause, String> {
        if self.0 == "geometry" {
            PointCase.new_geometry_query(field, &[LatLonGeometry::Polygon(Self::tiny())])
        } else {
            PointCase.new_geometry_query(field, geometries)
        }
    }
    fn quantize_lat(&self, lat: f64) -> f64 {
        quantize_lat(lat)
    }
    fn quantize_lon(&self, lon: f64) -> f64 {
        quantize_lon(lon)
    }
}

/// A case whose queries answer the wrong question fails the verify, for
/// boxes, circles, polygons and geometries; and the multi-valued test.
#[test]
fn verify_fails_on_a_wrong_query() {
    for (kind, names) in [
        ("box", "Rect"),
        ("circle", "radiusMeters"),
        ("polygon", "Polygon"),
        ("geometry", "["),
    ] {
        let err = std::panic::catch_unwind(|| {
            BaseGeoPoint::new(Wrong(kind), &mut JavaRandom::new(5), 1).random_medium();
        })
        .expect_err(kind);
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            msg.contains("some hits were wrong") && msg.contains(names),
            "{kind}: {msg}"
        );
    }
    let err = std::panic::catch_unwind(|| {
        BaseGeoPoint::new(Wrong("box"), &mut JavaRandom::new(5), 1).multi_valued();
    })
    .expect_err("multi-valued");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("some hits were wrong"), "{msg}");
}

/// A case that accepts what Java rejects fails the invalid-value tests.
#[test]
fn invalid_value_tests_fail_on_a_lenient_case() {
    struct Lenient;
    impl GeoPointCase for Lenient {
        fn add_point_to_doc(
            &self,
            _: &str,
            _: &mut Document,
            _: f64,
            _: f64,
        ) -> Result<(), String> {
            Ok(())
        }
        fn new_rect_query(&self, field: &str, _: &Rect) -> Result<Clause, String> {
            PointCase.new_distance_query(field, 0.0, 0.0, 1.0)
        }
        fn new_distance_query(&self, f: &str, _: f64, _: f64, _: f64) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
        fn new_polygon_query(&self, f: &str, _: &[Polygon]) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
        fn new_geometry_query(&self, f: &str, _: &[LatLonGeometry]) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
    }
    for which in 0..2 {
        let err = std::panic::catch_unwind(|| {
            let mut r = JavaRandom::new(5);
            let mut b = BaseGeoPoint::new(Lenient, &mut r, 1);
            if which == 0 {
                b.index_values();
            } else {
                b.invalid_queries();
            }
        })
        .expect_err("lenient");
        let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(
            msg.contains("expected an IllegalArgumentException"),
            "{msg}"
        );
    }
    // A message lacking the words Java requires.
    struct Terse;
    impl GeoPointCase for Terse {
        fn add_point_to_doc(
            &self,
            _: &str,
            _: &mut Document,
            _: f64,
            _: f64,
        ) -> Result<(), String> {
            Err("no".into())
        }
        fn new_rect_query(&self, _: &str, _: &Rect) -> Result<Clause, String> {
            Err("no".into())
        }
        fn new_distance_query(&self, _: &str, _: f64, _: f64, _: f64) -> Result<Clause, String> {
            Err("no".into())
        }
        fn new_polygon_query(&self, _: &str, _: &[Polygon]) -> Result<Clause, String> {
            Err("no".into())
        }
        fn new_geometry_query(&self, _: &str, _: &[LatLonGeometry]) -> Result<Clause, String> {
            Err("no".into())
        }
    }
    let err = std::panic::catch_unwind(|| {
        BaseGeoPoint::new(Terse, &mut JavaRandom::new(5), 1).invalid_queries();
    })
    .expect_err("terse");
    let msg = err.downcast_ref::<String>().cloned().unwrap_or_default();
    assert!(msg.contains("lacks"), "{msg}");
}

#[test]
fn rect_contains_point_is_inclusive_and_crosses_the_dateline() {
    let r = Rect {
        min_lat: -10.0,
        max_lat: 10.0,
        min_lon: -20.0,
        max_lon: 20.0,
    };
    assert!(rect_contains_point(&r, 10.0, -20.0));
    assert!(!rect_contains_point(&r, 10.5, 0.0));
    assert!(!rect_contains_point(&r, -10.5, 0.0));
    assert!(!rect_contains_point(&r, 0.0, 21.0));
    let crossing = Rect {
        min_lon: 170.0,
        max_lon: -170.0,
        ..r
    };
    assert!(rect_contains_point(&crossing, 0.0, 175.0));
    assert!(rect_contains_point(&crossing, 0.0, -175.0));
    assert!(!rect_contains_point(&crossing, 0.0, 0.0));
}

#[test]
#[should_panic(expected = "a missing point")]
fn rect_contains_point_rejects_nan() {
    let r = Rect {
        min_lat: 0.0,
        max_lat: 1.0,
        min_lon: 0.0,
        max_lon: 1.0,
    };
    let _ = rect_contains_point(&r, f64::NAN, 0.0);
}

/// `Math.nextAfter(int, double)` resolves to the float overload.
#[test]
fn next_after_is_javas_float_overload() {
    assert_eq!(java_next_after_f32(0.0, 90.0), f64::from(f32::from_bits(1)));
    assert_eq!(
        java_next_after_f32(1.0, -90.0),
        f64::from(1.0f32.next_down())
    );
    assert_eq!(java_next_after_f32(1.0, 1.0), 1.0);
}

#[test]
fn generators_stay_in_range() {
    let mut r = JavaRandom::new(11);
    assert_eq!(next_double_internal(&mut r, 3.0, 3.0), 3.0);
    for _ in 0..2000 {
        let v = next_double_internal(&mut r, -90.0, 90.0);
        assert!((-90.0..=90.0).contains(&v), "{v}");
        let b = next_box(&mut r);
        assert!(b.min_lat < b.max_lat);
        assert!(b.min_lon != b.max_lon);
        let b = next_box_not_crossing_dateline(&mut r);
        assert!(b.min_lon < b.max_lon);
        let (lat, lon, radius) = next_circle(&mut r);
        assert!((-90.0..=90.0).contains(&lat) && (-180.0..=180.0).contains(&lon));
        assert!(radius >= 1.0);
    }
}

#[test]
#[should_panic(expected = "low=2 high=1")]
fn next_double_internal_rejects_an_inverted_range() {
    let _ = next_double_internal(&mut JavaRandom::new(1), 2.0, 1.0);
}
