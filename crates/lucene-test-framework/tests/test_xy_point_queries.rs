//! Lucene's `BaseXYPointTestCase` subclasses `TestXYPointQueries` (BKD
//! points) and `TestXYDocValuesQueries` (the slow doc-values queries), over
//! every test method of the base with Java's verifiers.

use std::sync::Arc;

use lucene_index::document::{Document, XYDocValuesField, XYPointField};
use lucene_search::document::geo::{xy_doc_values_field, xy_point_field};
use lucene_search::document::DocumentQuery;
use lucene_search::extended_query::DocumentClause;
use lucene_search::query::Clause;
use lucene_test_framework::geo::base_xy_point::{BaseXYPoint, XYIndexed, XYQueries};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::geo::{XYGeometry, XYPolygon};
use lucene_util::java_random::JavaRandom;

fn keyed<Q: DocumentQuery + 'static>(
    key: String,
    q: lucene_search::Result<Q>,
) -> Result<Clause, String> {
    let q = q.map_err(|e| e.to_string())?;
    Ok(DocumentClause::new(key, Arc::new(q)).into())
}

/// `TestXYPointQueries`.
struct PointCase;

impl XYQueries for PointCase {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        x: f32,
        y: f32,
    ) -> Result<(), String> {
        doc.add(XYPointField::new(field, x, y).map_err(|e| e.to_string())?);
        Ok(())
    }
    fn new_rect_query(
        &self,
        field: &str,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    ) -> Result<Clause, String> {
        let q = xy_point_field::new_box_query(field, a, b, c, d);
        keyed(format!("box {field} {a} {b} {c} {d}"), q)
    }
    fn new_distance_query(&self, field: &str, x: f32, y: f32, r: f32) -> Result<Clause, String> {
        let q = xy_point_field::new_distance_query(field, x, y, r);
        keyed(format!("circle {field} {x} {y} {r}"), q)
    }
    fn new_polygon_query(&self, field: &str, polygons: &[XYPolygon]) -> Result<Clause, String> {
        let q = xy_point_field::new_polygon_query(field, polygons);
        keyed(format!("polygon {field} {polygons:?}"), q)
    }
    fn new_geometry_query(&self, field: &str, g: &[XYGeometry]) -> Result<Clause, String> {
        keyed(
            format!("geometry {field} {g:?}"),
            xy_point_field::new_geometry_query(field, g),
        )
    }
}

/// `TestXYDocValuesQueries`.
struct DocValuesCase;

impl XYQueries for DocValuesCase {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        x: f32,
        y: f32,
    ) -> Result<(), String> {
        doc.add(XYDocValuesField::new(field, x, y).map_err(|e| e.to_string())?);
        Ok(())
    }
    fn new_rect_query(
        &self,
        field: &str,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    ) -> Result<Clause, String> {
        let q = xy_doc_values_field::new_slow_box_query(field, a, b, c, d);
        keyed(format!("slow box {field} {a} {b} {c} {d}"), q)
    }
    fn new_distance_query(&self, field: &str, x: f32, y: f32, r: f32) -> Result<Clause, String> {
        let q = xy_doc_values_field::new_slow_distance_query(field, x, y, r);
        keyed(format!("slow circle {field} {x} {y} {r}"), q)
    }
    fn new_polygon_query(&self, field: &str, polygons: &[XYPolygon]) -> Result<Clause, String> {
        let q = xy_doc_values_field::new_slow_polygon_query(field, polygons);
        keyed(format!("slow polygon {field} {polygons:?}"), q)
    }
    fn new_geometry_query(&self, field: &str, g: &[XYGeometry]) -> Result<Clause, String> {
        let q = xy_doc_values_field::new_slow_geometry_query(field, g);
        keyed(format!("slow geometry {field} {g:?}"), q)
    }
}

type Method<Q> = fn(&mut BaseXYPoint<'_, XYIndexed<Q>>);

/// Every test method of the base, each on its own stream; the nightly ones
/// only under `tests.nightly`.
fn run_all<Q: XYQueries>(name: &str, make: impl Fn() -> Q) {
    let methods: [(&str, Method<Q>, bool); 15] = [
        ("random_tiny", |b| b.random_tiny().unwrap(), false),
        ("random_medium", |b| b.random_medium().unwrap(), false),
        ("random_big", |b| b.random_big().unwrap(), true),
        (
            "same_point_many_times",
            |b| b.same_point_many_times().unwrap(),
            false,
        ),
        ("low_cardinality", |b| b.low_cardinality().unwrap(), false),
        ("all_y_equal", |b| b.all_y_equal().unwrap(), false),
        ("all_x_equal", |b| b.all_x_equal().unwrap(), false),
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
    ];
    for (method, run, nightly) in methods {
        let mut tc = LuceneTestCase::new(&format!("{name}::{method}"));
        if nightly && !tc.nightly() {
            continue;
        }
        let multiplier = tc.multiplier();
        let mut case = XYIndexed {
            queries: make(),
            nightly: tc.nightly(),
        };
        run(&mut BaseXYPoint::new(&mut case, tc.random(), multiplier));
    }
}

#[test]
fn xy_point_queries() {
    run_all("TestXYPointQueries", || PointCase);
}

#[test]
fn xy_doc_values_queries() {
    run_all("TestXYDocValuesQueries", || DocValuesCase);
}

/// The nightly methods' code, once per run at a smaller size.
#[test]
fn nightly_methods_run() {
    let mut case = XYIndexed {
        queries: DocValuesCase,
        nightly: false,
    };
    BaseXYPoint::new(&mut case, &mut JavaRandom::new(3), 1).do_random_distance_test(2000, 10);
}

/// [`PointCase`] with one kind of query answering the wrong question.
struct Wrong(&'static str);

impl Wrong {
    fn tiny() -> XYPolygon {
        XYPolygon::new(
            &[0.0, 0.1, 0.1, 0.0, 0.0],
            &[0.0, 0.0, 0.1, 0.1, 0.0],
            vec![],
        )
        .unwrap()
    }
}

impl XYQueries for Wrong {
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        x: f32,
        y: f32,
    ) -> Result<(), String> {
        PointCase.add_point_to_doc(field, doc, x, y)
    }
    fn new_rect_query(
        &self,
        field: &str,
        a: f32,
        b: f32,
        c: f32,
        d: f32,
    ) -> Result<Clause, String> {
        if self.0 == "box" {
            PointCase.new_rect_query(field, -f32::MAX, f32::MAX, -f32::MAX, f32::MAX)
        } else {
            PointCase.new_rect_query(field, a, b, c, d)
        }
    }
    fn new_distance_query(&self, field: &str, x: f32, y: f32, r: f32) -> Result<Clause, String> {
        if self.0 == "circle" {
            PointCase.new_distance_query(field, 0.0, 0.0, 1.0)
        } else {
            PointCase.new_distance_query(field, x, y, r)
        }
    }
    fn new_polygon_query(&self, field: &str, polygons: &[XYPolygon]) -> Result<Clause, String> {
        if self.0 == "polygon" {
            PointCase.new_polygon_query(field, &[Self::tiny()])
        } else {
            PointCase.new_polygon_query(field, polygons)
        }
    }
    fn new_geometry_query(&self, field: &str, g: &[XYGeometry]) -> Result<Clause, String> {
        if self.0 == "geometry" {
            PointCase.new_geometry_query(field, &[XYGeometry::Polygon(Self::tiny())])
        } else {
            PointCase.new_geometry_query(field, g)
        }
    }
}

fn panic_message(f: impl FnOnce() + std::panic::UnwindSafe) -> String {
    let err = std::panic::catch_unwind(f).expect_err("the test should fail");
    err.downcast_ref::<String>().cloned().unwrap_or_default()
}

/// A case whose queries answer the wrong question fails the verifies, and
/// the multi-valued test.
#[test]
fn verify_fails_on_a_wrong_query() {
    for (kind, names) in [
        ("box", "XYRectangle"),
        ("circle", "radius"),
        ("polygon", "XYPolygon"),
        ("geometry", "["),
    ] {
        let msg = panic_message(|| {
            let mut case = XYIndexed {
                queries: Wrong(kind),
                nightly: false,
            };
            let _ = BaseXYPoint::new(&mut case, &mut JavaRandom::new(5), 1).random_medium();
        });
        assert!(
            msg.contains("some hits were wrong") && msg.contains(names),
            "{kind}: {msg}"
        );
    }
    let msg = panic_message(|| {
        let mut case = XYIndexed {
            queries: Wrong("box"),
            nightly: false,
        };
        BaseXYPoint::new(&mut case, &mut JavaRandom::new(5), 1).multi_valued();
    });
    assert!(msg.contains("some hits were wrong"), "{msg}");
}

/// A case that accepts what Java rejects fails the invalid-value tests.
#[test]
fn invalid_value_tests_fail_on_a_lenient_case() {
    struct Lenient;
    impl XYQueries for Lenient {
        fn add_point_to_doc(
            &self,
            _: &str,
            _: &mut Document,
            _: f32,
            _: f32,
        ) -> Result<(), String> {
            Ok(())
        }
        fn new_rect_query(
            &self,
            f: &str,
            _: f32,
            _: f32,
            _: f32,
            _: f32,
        ) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
        fn new_distance_query(&self, f: &str, _: f32, _: f32, _: f32) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
        fn new_polygon_query(&self, f: &str, _: &[XYPolygon]) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
        fn new_geometry_query(&self, f: &str, _: &[XYGeometry]) -> Result<Clause, String> {
            PointCase.new_distance_query(f, 0.0, 0.0, 1.0)
        }
    }
    for which in 0..2 {
        let msg = panic_message(|| {
            let mut case = XYIndexed {
                queries: Lenient,
                nightly: false,
            };
            let mut r = JavaRandom::new(5);
            let mut b = BaseXYPoint::new(&mut case, &mut r, 1);
            if which == 0 {
                b.index_values();
            } else {
                b.invalid_queries();
            }
        });
        assert!(
            msg.contains("expected an IllegalArgumentException"),
            "{msg}"
        );
    }
}
