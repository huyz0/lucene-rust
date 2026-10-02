//! Writes `GenGeoShapes`' corpus through this port's `IndexWriter` -- the
//! same documents (`fixtures/data/geo_shapes/docs.tsv`: `LatLonShape` and
//! `XYShape` triangles and shape doc values), commits and deletes -- for
//! `VerifyGeoShapes` to open with real Lucene 10.5.0: `CheckIndex`, then
//! every query of `queries.tsv` replayed through Lucene's own
//! `LatLonShape`/`XYShape` queries and the shape doc-values queries, which
//! must answer exactly as they answered over Lucene's own index.
//!
//! The triangles are seven-dimension points with four indexed dimensions,
//! so this is also the BKD writer's real-Lucene check for data dimensions
//! beyond the indexed ones.
//!
//! Usage: `write_geo_shapes_fixture <output-dir>`.
#![allow(clippy::arithmetic_side_effects, dead_code)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::{Document, LatLonShape, ShapeTriangle, Store, StringField, XYShape};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::document::geo::QueryRelation;
use lucene_store::FsDirectory;
use lucene_util::geo::{
    Circle, GeoError, LatLonGeometry, Line, Point, Polygon, Rectangle, XYCircle, XYGeometry,
    XYLine, XYPoint, XYPolygon, XYRectangle,
};

// ---------------------------------------------------------------- specs

fn nums(s: &str) -> Vec<f64> {
    s.split(',').map(|v| v.parse::<f64>().unwrap()).collect()
}

fn ring_d(s: &str) -> (Vec<f64>, Vec<f64>) {
    s.split(';')
        .map(|p| {
            let (x, y) = p.split_once(' ').unwrap();
            (x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap())
        })
        .unzip()
}

fn ring_f(s: &str) -> (Vec<f32>, Vec<f32>) {
    s.split(';')
        .map(|p| {
            let (x, y) = p.split_once(' ').unwrap();
            (x.parse::<f32>().unwrap(), y.parse::<f32>().unwrap())
        })
        .unzip()
}

/// A `GeoCorpus.spec` back into lat/lon geometries.
fn parse_latlon(spec: &str) -> Result<Vec<LatLonGeometry>, GeoError> {
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            Ok(match kind {
                "P:" => {
                    let v = nums(body);
                    LatLonGeometry::Point(Point::new(v[0], v[1])?)
                }
                "L:" => {
                    let (lats, lons) = ring_d(body);
                    LatLonGeometry::Line(Line::new(&lats, &lons)?)
                }
                "G:" => {
                    let mut rings = body.split('|');
                    let (lats, lons) = ring_d(rings.next().unwrap());
                    let holes = rings
                        .map(|r| {
                            let (la, lo) = ring_d(r);
                            Polygon::new(&la, &lo, vec![])
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    LatLonGeometry::Polygon(Polygon::new(&lats, &lons, holes)?)
                }
                "C:" => {
                    let v = nums(body);
                    LatLonGeometry::Circle(Circle::new(v[0], v[1], v[2])?)
                }
                "R:" => {
                    let v = nums(body);
                    LatLonGeometry::Rectangle(Rectangle::new(v[0], v[1], v[2], v[3])?)
                }
                other => panic!("{other}"),
            })
        })
        .collect()
}

fn parse_xy(spec: &str) -> Result<Vec<XYGeometry>, GeoError> {
    let fl = |s: &str| -> Vec<f32> { s.split(',').map(|v| v.parse().unwrap()).collect() };
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            Ok(match kind {
                "P:" => {
                    let v = fl(body);
                    XYGeometry::Point(XYPoint::new(v[0], v[1])?)
                }
                "L:" => {
                    let (x, y) = ring_f(body);
                    XYGeometry::Line(XYLine::new(&x, &y)?)
                }
                "G:" => {
                    let mut rings = body.split('|');
                    let (x, y) = ring_f(rings.next().unwrap());
                    let holes = rings
                        .map(|r| {
                            let (a, b) = ring_f(r);
                            XYPolygon::new(&a, &b, vec![])
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    XYGeometry::Polygon(XYPolygon::new(&x, &y, holes)?)
                }
                "C:" => {
                    let v = fl(body);
                    XYGeometry::Circle(XYCircle::new(v[0], v[1], v[2])?)
                }
                "R:" => {
                    let v = fl(body);
                    XYGeometry::Rectangle(XYRectangle::new(v[0], v[1], v[2], v[3])?)
                }
                other => panic!("{other}"),
            })
        })
        .collect()
}

fn relation(s: &str) -> QueryRelation {
    match s {
        "INTERSECTS" => QueryRelation::Intersects,
        "WITHIN" => QueryRelation::Within,
        "DISJOINT" => QueryRelation::Disjoint,
        "CONTAINS" => QueryRelation::Contains,
        other => panic!("{other}"),
    }
}

fn d(s: &str) -> f64 {
    s.parse().unwrap()
}

fn f(s: &str) -> f32 {
    s.parse().unwrap()
}

// ---------------------------------------------------------------- documents

/// `GenGeoShapes.fields`: a lat/lon shape's triangle fields.
fn latlon_fields(name: &str, g: &LatLonGeometry) -> Vec<ShapeTriangle> {
    match g {
        LatLonGeometry::Point(p) => LatLonShape::create_point_fields(name, p.lat(), p.lon()),
        LatLonGeometry::Line(l) => LatLonShape::create_line_fields(name, l),
        LatLonGeometry::Polygon(p) => LatLonShape::create_indexable_fields(name, p),
        other => panic!("not a shape: {other:?}"),
    }
    .unwrap()
}

fn xy_fields(name: &str, g: &XYGeometry) -> Vec<ShapeTriangle> {
    match g {
        XYGeometry::Point(p) => XYShape::create_point_fields(name, p.x(), p.y()),
        XYGeometry::Line(l) => XYShape::create_line_fields(name, l),
        XYGeometry::Polygon(p) => XYShape::create_indexable_fields(name, p),
        other => panic!("not a shape: {other:?}"),
    }
    .unwrap()
}

/// `GenGeoShapes.latLonDocValue`: Java's way to one shape's doc value.
fn latlon_doc_value(
    name: &str,
    g: &LatLonGeometry,
) -> lucene_index::document::LatLonShapeDocValuesField {
    match g {
        LatLonGeometry::Point(p) => {
            LatLonShape::create_point_doc_value_field(name, p.lat(), p.lon())
        }
        LatLonGeometry::Line(l) => LatLonShape::create_line_doc_value_field(name, l),
        LatLonGeometry::Polygon(p) => LatLonShape::create_doc_value_field(name, p),
        other => panic!("not a shape: {other:?}"),
    }
    .unwrap()
}

fn xy_doc_value(name: &str, g: &XYGeometry) -> lucene_index::document::XYShapeDocValuesField {
    match g {
        XYGeometry::Point(p) => XYShape::create_point_doc_value_field(name, p.x(), p.y()),
        XYGeometry::Line(l) => XYShape::create_line_doc_value_field(name, l),
        XYGeometry::Polygon(p) => XYShape::create_doc_value_field(name, p),
        other => panic!("not a shape: {other:?}"),
    }
    .unwrap()
}

/// `GenGeoShapes.document`: `id`, every shape's triangles in line order,
/// then the `shape` doc value, then the `xy` one.
fn document(line: &str) -> Document {
    let p: Vec<&str> = line.split('\t').collect();
    let mut doc = Document::new();
    doc.add(StringField::new("id", p[0], Store::Yes));
    let (mut shape, mut shapes) = (Vec::new(), Vec::new());
    let (mut xy, mut xys) = (Vec::new(), Vec::new());
    for pair in p[1..].chunks(2) {
        let (field, spec) = (pair[0], pair[1]);
        if field == "xy" {
            let g = parse_xy(spec).unwrap().remove(0);
            let fs = xy_fields(field, &g);
            for t in &fs {
                doc.add(t.clone());
            }
            xy.extend(fs);
            xys.push(g);
        } else {
            let g = parse_latlon(spec).unwrap().remove(0);
            let fs = latlon_fields(field, &g);
            for t in &fs {
                doc.add(t.clone());
            }
            if field == "shape" {
                shape.extend(fs);
                shapes.push(g);
            }
        }
    }
    match shapes.len() {
        0 => {}
        1 => doc.add(latlon_doc_value("shape", &shapes[0])),
        _ => doc.add(LatLonShape::create_doc_value_field_from_fields("shape", &shape).unwrap()),
    }
    match xys.len() {
        0 => {}
        1 => doc.add(xy_doc_value("xy", &xys[0])),
        _ => doc.add(XYShape::create_doc_value_field_from_fields("xy", &xy).unwrap()),
    }
    doc
}

fn main() {
    let out = std::env::args()
        .nth(1)
        .expect("usage: write_geo_shapes_fixture <output-dir>");
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo_shapes");
    let read = |f: &str| std::fs::read_to_string(fixtures.join(f)).expect("GenGeoShapes fixtures");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).expect("create output dir");
    let fs = FsDirectory::open(std::path::Path::new(&out));
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).expect("open writer");
    for (i, line) in read("docs.tsv").lines().enumerate() {
        w.add_fields_document(&document(line)).expect("add");
        if matches!(i + 1, 500 | 1000 | 1500 | 1540) {
            w.commit().expect("commit");
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)])
            .expect("delete");
    }
    w.commit().expect("commit");
}
