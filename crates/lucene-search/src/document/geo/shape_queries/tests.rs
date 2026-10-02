//! The shape queries' own edges over small indexes this port writes; their
//! agreement with Lucene is `tests/geo_shapes_fixtures.rs`.

use lucene_index::document::{self as d, Document, IndexableField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_store::FsDirectory;
use lucene_util::geo::{Circle, Line, Point, Polygon, XYCircle, XYLine, XYPoint, XYPolygon};
use lucene_util::test_support::TempDir;

use super::super::{lat_lon_shape, xy_shape, MustConjunction};
use super::*;
use crate::directory_reader::DirectoryReader;
use crate::document::search_all;

const VERSION: LuceneVersion = LuceneVersion {
    major: 10,
    minor: 5,
    bugfix: 0,
};

fn index(tmp: &TempDir, docs: Vec<Vec<Box<dyn IndexableField>>>) -> DirectoryReader {
    let dir = FsDirectory::open(tmp.path());
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", VERSION).unwrap();
    for fields in docs {
        let mut doc = Document::new();
        for f in fields {
            doc.add_boxed(f);
        }
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
    drop(w);
    DirectoryReader::open(&dir).unwrap()
}

fn square(lat: f64, lon: f64, w: f64) -> Polygon {
    Polygon::new(
        &[lat, lat, lat + w, lat + w, lat],
        &[lon, lon + w, lon + w, lon, lon],
        vec![],
    )
    .unwrap()
}

/// A polygon's triangles and its doc value under `field`.
fn shape(field: &str, p: &Polygon) -> Vec<Box<dyn IndexableField>> {
    let mut out: Vec<Box<dyn IndexableField>> = d::LatLonShape::create_indexable_fields(field, p)
        .unwrap()
        .into_iter()
        .map(|t| Box::new(t) as Box<dyn IndexableField>)
        .collect();
    out.push(Box::new(
        d::LatLonShape::create_doc_value_field(field, p).unwrap(),
    ));
    out
}

fn hits(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> Vec<(i32, f32)> {
    search_all(leaves, q)
        .unwrap()
        .into_iter()
        .map(|h| (h.doc_id, h.score))
        .collect()
}

#[test]
fn encoded_rectangle_relations() {
    let r = EncodedRectangle {
        min_x: 0,
        max_x: 10,
        min_y: 0,
        max_y: 10,
        wraps_coordinate_system: false,
    };
    let w = EncodedRectangle {
        min_x: 10,
        max_x: 0,
        wraps_coordinate_system: true,
        ..r
    };
    assert!(r.contains(5, 5) && !r.contains(11, 5) && !r.contains(5, 11));
    assert!(w.contains(-5, 5) && w.contains(15, 5) && !w.contains(5, 5));
    // lines
    assert!(r.intersects_line(5, 5, 50, 50));
    assert!(r.intersects_line(-5, 5, 15, 5), "crossing through");
    assert!(!r.intersects_line(-5, 20, 15, 20), "above");
    assert!(!r.intersects_line(20, 5, 30, 5), "beside");
    assert!(!r.intersects_line(-5, 9, 1, 15), "near a corner");
    assert!(w.intersects_line(5, 5, 15, 5));
    assert!(
        !w.intersects_line(2, 5, 8, 5),
        "in the gap of a wrapping box"
    );
    // triangles
    assert!(r.intersects_triangle(5, 5, 50, 50, 50, 0));
    assert!(
        r.intersects_triangle(-10, -10, 30, -10, -10, 30),
        "containing the box"
    );
    assert!(!r.intersects_triangle(-10, 20, 30, 20, -10, 30));
    assert!(!r.intersects_triangle(20, 0, 30, 0, 20, 10));
    assert!(
        r.intersects_triangle(-5, 5, 5, -5, 15, 15),
        "an edge through"
    );
    assert!(!w.intersects_triangle(2, 2, 8, 2, 5, 8));
    assert!(w.intersects_triangle(2, 2, 12, 2, 5, 8));
    // rectangles
    assert!(r.intersects_rectangle(5, 15, 5, 15));
    assert!(!r.intersects_rectangle(5, 15, 11, 15));
    assert!(!r.intersects_rectangle(11, 15, 5, 15));
    assert!(w.intersects_rectangle(2, 3, 5, 6));
    assert!(r.contains_rectangle(1, 9, 1, 9) && !r.contains_rectangle(1, 11, 1, 9));
    // containment
    assert!(r.contains_line(1, 1, 9, 9) && !r.contains_line(1, 1, 11, 9));
    assert!(!r.contains_line(1, -1, 9, 9));
    assert!(w.contains_line(11, 1, 12, 9) && w.contains_line(-3, 1, -1, 9));
    assert!(!w.contains_line(-3, 1, 12, 9));
    assert!(r.contains_triangle(1, 1, 9, 1, 5, 9));
    assert!(!r.contains_triangle(1, 1, 9, 1, 5, 19));
    assert!(w.contains_triangle(11, 1, 19, 1, 15, 9));
    assert!(!w.contains_triangle(-1, 1, 19, 1, 15, 9));
    // within
    assert_eq!(r.within_line(5, 5, true, 50, 50), WithinRelation::NotWithin);
    assert_eq!(r.within_line(-5, 5, true, 15, 5), WithinRelation::NotWithin);
    assert_eq!(r.within_line(-5, 5, false, 15, 5), WithinRelation::Disjoint);
    assert_eq!(
        r.within_triangle(5, 5, true, 50, 0, true, 50, 50, true),
        WithinRelation::NotWithin
    );
    assert_eq!(
        r.within_triangle(20, 20, true, 30, 20, true, 20, 30, true),
        WithinRelation::Disjoint
    );
    assert_eq!(
        r.within_triangle(20, 0, true, 30, 0, true, 20, 10, true),
        WithinRelation::Disjoint
    );
    // the triangle contains the box: a candidate
    assert_eq!(
        r.within_triangle(-10, -10, true, 40, -10, true, -10, 40, true),
        WithinRelation::Candidate
    );
    // an edge through the box: not within when it is the shape's, a
    // candidate when it is internal to the tessellation
    for (ab, bc, ca, want) in [
        (true, false, false, WithinRelation::NotWithin),
        (false, false, false, WithinRelation::Candidate),
    ] {
        assert_eq!(r.within_triangle(-5, -5, ab, 15, 5, bc, -5, 50, ca), want);
    }
    // the same edge as `bc`, then as `ca`
    assert_eq!(
        r.within_triangle(-5, 50, false, -5, -5, true, 15, 5, false),
        WithinRelation::NotWithin
    );
    assert_eq!(
        r.within_triangle(15, 5, false, -5, 50, false, -5, -5, true),
        WithinRelation::NotWithin
    );
    assert_eq!(
        r.within_triangle(15, 5, false, -5, 50, false, -5, -5, false),
        WithinRelation::Candidate
    );
    assert_eq!(
        w.within_triangle(2, 20, true, 8, 20, true, 5, 30, true),
        WithinRelation::Disjoint
    );
    assert_eq!(
        w.within_triangle(2, 2, false, 8, 2, false, 5, 8, false),
        WithinRelation::Disjoint
    );
}

#[test]
fn shape_queries_end_to_end() {
    let tmp = TempDir::new("geo-shape-queries");
    let mut docs = vec![
        shape("s", &square(0.0, 0.0, 10.0)),
        shape("s", &square(20.0, 20.0, 1.0)),
    ];
    // a non-shape point field and a binary doc value that is no shape
    docs.push(vec![
        Box::new(d::IntPoint::new("int", &[1]).unwrap()),
        Box::new(d::BinaryDocValuesField::new("junk", vec![0u8, 1, 2])),
        Box::new(d::NumericDocValuesField::new("num", 3)),
    ]);
    // next to the dateline
    docs.push(shape("s", &square(0.0, 179.0, 0.5)));
    let r = index(&tmp, docs);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();

    // the factories
    let q =
        lat_lon_shape::new_polygon_query("s", QueryRelation::Intersects, &[square(5.0, 5.0, 1.0)])
            .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(0, 1.0)]);
    let line = Line::new(&[19.0, 22.0], &[19.0, 22.0]).unwrap();
    let q = lat_lon_shape::new_line_query("s", QueryRelation::Intersects, &[line]).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(1, 1.0)]);
    let q = lat_lon_shape::new_point_query("s", QueryRelation::Contains, &[[5.0, 5.0]]).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(0, 1.0)]);
    assert!(lat_lon_shape::new_point_query("s", QueryRelation::Contains, &[[95.0, 5.0]]).is_err());
    let c = Circle::new(20.5, 20.5, 1000.0).unwrap();
    let q = lat_lon_shape::new_distance_query("s", QueryRelation::Within, &[c]).unwrap();
    assert!(hits(&leaves, q.as_ref()).is_empty());
    let q =
        lat_lon_shape::new_box_query("s", QueryRelation::Within, -1.0, 30.0, -1.0, 30.0).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()).len(), 2);
    // a CONTAINS of several geometries: one constant-score conjunction
    let g = [
        LatLonGeometry::Point(Point::new(1.0, 1.0).unwrap()),
        LatLonGeometry::Rectangle(Rectangle::new(2.0, 3.0, 2.0, 3.0).unwrap()),
    ];
    let q = lat_lon_shape::new_geometry_query("s", QueryRelation::Contains, &g).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(0, 1.0)]);
    // a CONTAINS box across the dateline: two clauses whose scores add up
    let q = lat_lon_shape::new_box_query("s", QueryRelation::Contains, 0.0, 1.0, 179.0, -179.0)
        .unwrap();
    assert!(hits(&leaves, q.as_ref()).is_empty());
    let q = lat_lon_shape::new_slow_doc_values_box_query(
        "s",
        QueryRelation::Contains,
        0.0,
        1.0,
        179.0,
        -179.0,
    )
    .unwrap();
    assert!(hits(&leaves, q.as_ref()).is_empty());
    let both = MustConjunction {
        clauses: vec![
            lat_lon_shape::new_box_query("s", QueryRelation::Intersects, 0.0, 1.0, 0.0, 1.0)
                .unwrap(),
            lat_lon_shape::new_box_query("s", QueryRelation::Intersects, 0.0, 2.0, 0.0, 2.0)
                .unwrap(),
        ],
    };
    let got = search_all(&leaves, &crate::document::Boosted::new(Box::new(both), 1.5)).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].score, 3.0);

    // the doc-values form
    let q = lat_lon_shape::new_slow_doc_values_box_query(
        "s",
        QueryRelation::Intersects,
        19.0,
        25.0,
        19.0,
        25.0,
    )
    .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(1, 1.0)]);
    let e = lat_lon_shape::new_slow_doc_values_box_query(
        "s",
        QueryRelation::Contains,
        0.0,
        1.0,
        0.0,
        1.0,
    )
    .unwrap_err();
    assert!(
        e.to_string().contains("does not yet support CONTAINS"),
        "{e}"
    );
    let dvq = LatLonShapeDocValuesQuery::new(
        "s",
        QueryRelation::Disjoint,
        &[LatLonGeometry::Point(Point::new(5.0, 5.0).unwrap())],
    )
    .unwrap();
    assert_eq!(
        (dvq.field(), dvq.query_relation()),
        ("s", QueryRelation::Disjoint)
    );
    assert_eq!(hits(&leaves, &dvq), vec![(1, 1.0), (3, 1.0)]);
    // a field without binary doc values, or no such field: nothing
    for field in ["num", "nope"] {
        let q = LatLonShapeDocValuesQuery::new(
            field,
            QueryRelation::Intersects,
            &[LatLonGeometry::Point(Point::new(5.0, 5.0).unwrap())],
        )
        .unwrap();
        assert!(hits(&leaves, &q).is_empty());
    }
    // a binary doc value that is not a shape is an error, not a panic
    let q = LatLonShapeDocValuesQuery::new(
        "junk",
        QueryRelation::Intersects,
        &[LatLonGeometry::Point(Point::new(5.0, 5.0).unwrap())],
    )
    .unwrap();
    assert!(search_all(&leaves, &q).is_err());
    let q = XYShapeDocValuesQuery::new(
        "junk",
        QueryRelation::Intersects,
        &[XYGeometry::Point(XYPoint::new(5.0, 5.0).unwrap())],
    )
    .unwrap();
    assert!(search_all(&leaves, &q).is_err());
    assert_eq!(
        (q.field(), q.query_relation()),
        ("junk", QueryRelation::Intersects)
    );

    // a point field that is not a shape's
    let e = search_all(
        &leaves,
        lat_lon_shape::new_box_query("int", QueryRelation::Intersects, 0.0, 1.0, 0.0, 1.0)
            .unwrap()
            .as_ref(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("not a shape's"), "{e}");

    // a line under WITHIN, and a box across the dateline under CONTAINS
    // built directly (Lucene's factories split it)
    let line = Line::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap();
    assert!(
        LatLonShapeQuery::new("s", QueryRelation::Within, &[LatLonGeometry::Line(line)]).is_err()
    );
    let q = LatLonShapeBoundingBoxQuery::new(
        "s",
        QueryRelation::Contains,
        Rectangle::new(0.0, 1.0, 179.0, -179.0).unwrap(),
    )
    .unwrap();
    let e = search_all(&leaves, &q).unwrap_err();
    assert!(e.to_string().contains("crossing the date line"), "{e}");
    // and from the dateline itself: -180 is its west edge
    let q = LatLonShapeBoundingBoxQuery::new(
        "s",
        QueryRelation::Intersects,
        Rectangle::new(0.0, 30.0, 180.0, 30.0).unwrap(),
    )
    .unwrap();
    assert_eq!(hits(&leaves, &q).len(), 2);
}

#[test]
fn xy_shape_queries_end_to_end() {
    let tmp = TempDir::new("geo-xy-shape-queries");
    let poly = XYPolygon::new(
        &[0.0, 10.0, 10.0, 0.0, 0.0],
        &[0.0, 0.0, 10.0, 10.0, 0.0],
        vec![],
    )
    .unwrap();
    let mut fields: Vec<Box<dyn IndexableField>> = d::XYShape::create_indexable_fields("xy", &poly)
        .unwrap()
        .into_iter()
        .map(|t| Box::new(t) as Box<dyn IndexableField>)
        .collect();
    fields.push(Box::new(
        d::XYShape::create_doc_value_field("xy", &poly).unwrap(),
    ));
    let line = XYLine::new(&[20.0, 30.0], &[20.0, 30.0]).unwrap();
    let mut second: Vec<Box<dyn IndexableField>> = d::XYShape::create_line_fields("xy", &line)
        .unwrap()
        .into_iter()
        .map(|t| Box::new(t) as Box<dyn IndexableField>)
        .collect();
    second.push(Box::new(
        d::XYShape::create_line_doc_value_field("xy", &line).unwrap(),
    ));
    let r = index(&tmp, vec![fields, second]);
    let opened = r.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let q = xy_shape::new_box_query("xy", QueryRelation::Intersects, 1.0, 2.0, 1.0, 2.0).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(0, 1.0)]);
    let q = xy_shape::new_slow_doc_values_box_query(
        "xy",
        QueryRelation::Within,
        15.0,
        35.0,
        15.0,
        35.0,
    )
    .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(1, 1.0)]);
    assert!(xy_shape::new_slow_doc_values_box_query(
        "xy",
        QueryRelation::Contains,
        0.0,
        1.0,
        0.0,
        1.0
    )
    .is_err());
    let q =
        xy_shape::new_line_query("xy", QueryRelation::Within, std::slice::from_ref(&line)).unwrap();
    assert!(hits(&leaves, q.as_ref()).is_empty());
    let q = xy_shape::new_polygon_query("xy", QueryRelation::Disjoint, std::slice::from_ref(&poly))
        .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(1, 1.0)]);
    let q = xy_shape::new_point_query("xy", QueryRelation::Contains, &[[1.0, 1.0], [2.0, 2.0]])
        .unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(0, 1.0)]);
    assert!(xy_shape::new_point_query("xy", QueryRelation::Contains, &[[f32::NAN, 1.0]]).is_err());
    let c = XYCircle::new(25.0, 25.0, 1.0).unwrap();
    let q = xy_shape::new_distance_query("xy", QueryRelation::Intersects, &[c]).unwrap();
    assert_eq!(hits(&leaves, q.as_ref()), vec![(1, 1.0)]);
    let q = XYShapeQuery::new("xy", QueryRelation::Intersects, &[XYGeometry::Line(line)]).unwrap();
    assert_eq!(q.geometries.len(), 1);
}
