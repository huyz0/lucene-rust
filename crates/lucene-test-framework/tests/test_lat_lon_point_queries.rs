//! Lucene's `BaseGeoPointTestCase` subclasses `TestLatLonPointQueries`
//! (BKD points) and `TestLatLonDocValuesQueries` (the slow doc-values
//! queries), over the random tests the base ports; and `GeoTestUtil`'s
//! generators.

use std::sync::Arc;

use lucene_index::document::{Document, LatLonDocValuesField, LatLonPoint};
use lucene_search::document::geo::{lat_lon_doc_values_field, lat_lon_point};
use lucene_search::document::DocumentQuery;
use lucene_search::extended_query::DocumentClause;
use lucene_search::query::Clause;
use lucene_test_framework::geo::base_geo_point::{rect_contains_point, BaseGeoPoint, GeoPointCase};
use lucene_test_framework::geo::geo_test_util::{
    java_next_after_f32, next_box, next_box_not_crossing_dateline, next_circle,
    next_double_internal, Rect,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::geo::GeoEncodingUtils;
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

/// `TestLatLonPointQueries`.
struct PointCase;

impl GeoPointCase for PointCase {
    fn add_point_to_doc(&self, field: &str, doc: &mut Document, lat: f64, lon: f64) {
        doc.add(LatLonPoint::new(field, lat, lon).expect("a valid point"));
    }

    fn new_rect_query(&self, field: &str, r: &Rect) -> Clause {
        let q = lat_lon_point::new_box_query(field, r.min_lat, r.max_lat, r.min_lon, r.max_lon);
        clause(format!("box {r:?}"), q.expect("a valid box"))
    }

    fn new_distance_query(&self, field: &str, lat: f64, lon: f64, radius: f64) -> Clause {
        let q = lat_lon_point::new_distance_query(field, lat, lon, radius);
        clause(
            format!("circle {lat} {lon} {radius}"),
            q.expect("a valid circle"),
        )
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
    fn add_point_to_doc(&self, field: &str, doc: &mut Document, lat: f64, lon: f64) {
        doc.add(LatLonDocValuesField::new(field, lat, lon).expect("a valid point"));
    }

    fn new_rect_query(&self, field: &str, r: &Rect) -> Clause {
        let q = lat_lon_doc_values_field::new_slow_box_query(
            field, r.min_lat, r.max_lat, r.min_lon, r.max_lon,
        );
        clause(format!("slow box {r:?}"), q.expect("a valid box"))
    }

    fn new_distance_query(&self, field: &str, lat: f64, lon: f64, radius: f64) -> Clause {
        let q = lat_lon_doc_values_field::new_slow_distance_query(field, lat, lon, radius);
        clause(
            format!("slow circle {lat} {lon} {radius}"),
            q.expect("a valid circle"),
        )
    }

    fn quantize_lat(&self, lat: f64) -> f64 {
        quantize_lat(lat)
    }

    fn quantize_lon(&self, lon: f64) -> f64 {
        quantize_lon(lon)
    }
}

/// Every random test method of the base, each on its own stream.
fn run_all<C: GeoPointCase>(name: &str, make: impl Fn() -> C) {
    type Method<C> = fn(&mut BaseGeoPoint<'_, C>);
    let methods: [(&str, Method<C>); 6] = [
        ("random_tiny", |b| b.random_tiny()),
        ("random_medium", |b| b.random_medium()),
        ("same_point_many_times", |b| b.same_point_many_times()),
        ("low_cardinality", |b| b.low_cardinality()),
        ("all_lat_equal", |b| b.all_lat_equal()),
        ("all_lon_equal", |b| b.all_lon_equal()),
    ];
    for (method, run) in methods {
        let mut tc = LuceneTestCase::new(&format!("{name}::{method}"));
        let multiplier = tc.multiplier();
        run(&mut BaseGeoPoint::new(make(), tc.random(), multiplier));
    }
}

#[test]
fn lat_lon_point_queries() {
    run_all("TestLatLonPointQueries", || PointCase);
}

#[test]
fn lat_lon_doc_values_queries() {
    run_all("TestLatLonDocValuesQueries", || DocValuesCase);
}

/// A case whose queries answer the wrong question fails the verify, for
/// boxes and for circles.
#[test]
fn verify_fails_on_a_wrong_query() {
    /// Boxes answer with the whole world.
    struct WrongBox;
    impl GeoPointCase for WrongBox {
        fn add_point_to_doc(&self, field: &str, doc: &mut Document, lat: f64, lon: f64) {
            PointCase.add_point_to_doc(field, doc, lat, lon);
        }
        fn new_rect_query(&self, field: &str, _: &Rect) -> Clause {
            let world = Rect {
                min_lat: -90.0,
                max_lat: 90.0,
                min_lon: -180.0,
                max_lon: 180.0,
            };
            PointCase.new_rect_query(field, &world)
        }
        fn new_distance_query(&self, field: &str, lat: f64, lon: f64, radius: f64) -> Clause {
            PointCase.new_distance_query(field, lat, lon, radius)
        }
    }
    /// Circles answer with a one-metre circle at the origin.
    struct WrongCircle;
    impl GeoPointCase for WrongCircle {
        fn add_point_to_doc(&self, field: &str, doc: &mut Document, lat: f64, lon: f64) {
            PointCase.add_point_to_doc(field, doc, lat, lon);
        }
        fn new_rect_query(&self, field: &str, r: &Rect) -> Clause {
            PointCase.new_rect_query(field, r)
        }
        fn new_distance_query(&self, field: &str, _: f64, _: f64, _: f64) -> Clause {
            PointCase.new_distance_query(field, 0.0, 0.0, 1.0)
        }
        fn quantize_lat(&self, lat: f64) -> f64 {
            quantize_lat(lat)
        }
        fn quantize_lon(&self, lon: f64) -> f64 {
            quantize_lon(lon)
        }
    }
    let box_err = std::panic::catch_unwind(|| {
        BaseGeoPoint::new(WrongBox, &mut JavaRandom::new(5), 1).random_medium();
    })
    .expect_err("the world is not every box");
    let msg = box_err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        msg.contains("some hits were wrong") && msg.contains("Rect"),
        "{msg}"
    );

    let circle_err = std::panic::catch_unwind(|| {
        BaseGeoPoint::new(WrongCircle, &mut JavaRandom::new(5), 1).random_medium();
    })
    .expect_err("a one-metre circle is not every circle");
    let msg = circle_err
        .downcast_ref::<String>()
        .cloned()
        .unwrap_or_default();
    assert!(
        msg.contains("some hits were wrong") && msg.contains("radiusMeters"),
        "{msg}"
    );
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
