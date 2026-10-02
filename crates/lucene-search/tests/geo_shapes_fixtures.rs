//! The shape fields, their doc values and the shape queries, differentially
//! against Lucene 10.5.0: `fixtures/src/GenGeoShapes.java` indexed a seeded
//! corpus of `LatLonShape`/`XYShape` shapes (points, lines, polygons with
//! holes, several shapes per document, the poles and the dateline, slivers,
//! collinear runs, duplicates, documents without the field, four segments
//! with deletions) with their shape doc values, and recorded its answer to
//! ~1600 queries -- every geometry under every relation, indexed and over the
//! doc values. This builds the same queries with
//! `lucene_search::document::geo` and requires the same hits and scores, over
//! Java's index and over an index this port writes from the same documents,
//! whose triangles and doc values must also be Lucene's byte for byte. It
//! also checks `ShapeField`'s triangle encoding and the `ShapeDocValues`
//! tree (bytes, header, centroid, bounding box, `relate`) on their own.

#![allow(clippy::arithmetic_side_effects)]

use std::borrow::Cow;

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, LatLonShape, LatLonShapeDocValues, ShapeField, ShapeTriangle, Store, StringField,
    XYShape, XYShapeDocValues,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::{
    lat_lon_shape, xy_shape, LatLonShapeDocValuesQuery, QueryRelation, XYShapeDocValuesQuery,
};
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::{
    Circle, GeoError, LatLonGeometry, Line, Point, Polygon, Rectangle, XYCircle, XYGeometry,
    XYLine, XYPoint, XYPolygon, XYRectangle,
};
use lucene_util::test_support::TempDir;

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo_shapes")
}

fn read(file: &str) -> String {
    std::fs::read_to_string(root().join(file))
        .expect("run scripts/gen-fixtures.sh --only GenGeoShapes")
}

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

/// The same documents, written by this port: every commit and delete
/// `GenGeoShapes.main` made, in order.
fn write_rust_index(dir: &std::path::Path) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for (i, line) in read("docs.tsv").lines().enumerate() {
        w.add_fields_document(&document(line)).unwrap();
        // GenGeoShapes commits after each 500-document segment and the
        // 40-document one.
        if matches!(i + 1, 500 | 1000 | 1500 | 1540) {
            w.commit().unwrap();
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)]).unwrap();
    }
    w.commit().unwrap();
}

// ---------------------------------------------------------------- answers

fn hex_bits(hits: &[i32]) -> String {
    let max = hits.iter().copied().max().unwrap_or(-1);
    let mut b = vec![0u8; ((max + 8) / 8) as usize];
    for &d in hits {
        b[(d >> 3) as usize] |= 1 << (d & 7);
    }
    while b.last() == Some(&0) {
        b.pop();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// `VerifyGeoShapes.constant`: `C total hexbits scorebits` or `E message`.
fn constant(
    segments: &[OpenSegment<'_>],
    q: Result<Box<dyn DocumentQuery>, lucene_search::Error>,
) -> String {
    let error = |e: lucene_search::Error| {
        let msg = e.to_string();
        format!("E\t{}", msg.split_once(": ").map_or(&*msg, |m| m.1))
    };
    let q = match q {
        Ok(q) => q,
        Err(e) => return error(e),
    };
    let hits = match dq::search_all(segments, q.as_ref()) {
        Ok(h) => h,
        Err(e) => return error(e),
    };
    let score = hits.first().map_or(0.0, |h| h.score);
    assert!(hits.iter().all(|h| h.score == score), "not constant");
    let ids: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
    format!(
        "C\t{}\t{}\t{:x}",
        ids.len(),
        hex_bits(&ids),
        score.to_bits()
    )
}

fn bx<Q: DocumentQuery + 'static>(
    q: Result<Q, lucene_search::Error>,
) -> Result<Box<dyn DocumentQuery>, lucene_search::Error> {
    q.map(|q| Box::new(q) as Box<dyn DocumentQuery>)
}

/// `VerifyGeoShapes.run`.
fn run(segments: &[OpenSegment<'_>], a: &[&str]) -> String {
    let field = a[1];
    let rel = relation(a[2]);
    let q = match a[0] {
        "geom" => lat_lon_shape::new_geometry_query(field, rel, &parse_latlon(a[3]).unwrap()),
        "dvgeom" => bx(LatLonShapeDocValuesQuery::new(
            field,
            rel,
            &parse_latlon(a[3]).unwrap(),
        )),
        "box" => lat_lon_shape::new_box_query(field, rel, d(a[3]), d(a[4]), d(a[5]), d(a[6])),
        "dvbox" => lat_lon_shape::new_slow_doc_values_box_query(
            field,
            rel,
            d(a[3]),
            d(a[4]),
            d(a[5]),
            d(a[6]),
        ),
        "xygeom" => xy_shape::new_geometry_query(field, rel, &parse_xy(a[3]).unwrap()),
        "xydvgeom" => bx(XYShapeDocValuesQuery::new(
            field,
            rel,
            &parse_xy(a[3]).unwrap(),
        )),
        "xybox" => xy_shape::new_box_query(field, rel, f(a[3]), f(a[4]), f(a[5]), f(a[6])),
        "xydvbox" => {
            xy_shape::new_slow_doc_values_box_query(field, rel, f(a[3]), f(a[4]), f(a[5]), f(a[6]))
        }
        other => panic!("unknown query {other}"),
    };
    constant(segments, q)
}

/// Every query against `dir`; returns how many ran.
fn check_queries(dir: &std::path::Path) -> usize {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 4, "segments");
    let mut failures = Vec::new();
    let mut n = 0;
    for line in read("queries.tsv").lines() {
        let (query, want) = line.split_once("\t=>\t").unwrap();
        let a: Vec<&str> = query.split('\t').collect();
        let got = run(&segments, &a);
        if got != want {
            let short = |s: &str| s.chars().take(240).collect::<String>();
            failures.push(format!(
                "{}\n  java: {}\n  rust: {}",
                short(query),
                short(want),
                short(&got)
            ));
        }
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} queries differ:\n{}",
        failures.len(),
        failures[..failures.len().min(25)].join("\n")
    );
    n
}

#[test]
fn every_shape_query_matches_lucene_on_lucenes_index() {
    let n = check_queries(&root().join("index"));
    assert!(n > 1500, "{n} queries");
}

#[test]
fn every_shape_query_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("geo-shapes-write");
    write_rust_index(tmp.path());
    let n = check_queries(tmp.path());
    assert!(n > 1500, "{n} queries");
}

#[test]
fn every_triangle_and_doc_value_is_written_as_lucene_writes_it() {
    // The packed triangles and the binary doc values of the two indexes,
    // segment by segment and document by document.
    let tmp = TempDir::new("geo-shapes-bytes");
    write_rust_index(tmp.path());
    let dump = |dir: &std::path::Path| -> Vec<String> {
        let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
        let mut out = Vec::new();
        for r in reader.segment_readers() {
            let points = r.points_reader().unwrap();
            let mut infos = r.field_infos().fields.clone();
            infos.sort_by(|a, b| a.name.cmp(&b.name));
            for info in infos {
                out.push(format!(
                    "{} points={}/{}/{} dv={:?}",
                    info.name,
                    info.point_dimension_count,
                    info.point_index_dimension_count,
                    info.point_num_bytes,
                    info.doc_values_type
                ));
                if let Some(pf) = points.field(info.number) {
                    let mut all: Vec<(i32, Vec<u8>)> = points
                        .decode_all_points(info.number)
                        .unwrap()
                        .into_iter()
                        .map(|p| (p.doc_id, p.packed_value))
                        .collect();
                    all.sort();
                    out.push(format!(
                        "  {} points, {} docs, {:?}..{:?}",
                        pf.point_count, pf.doc_count, pf.min_packed_value, pf.max_packed_value,
                    ));
                    out.extend(all.iter().map(|(d, v)| format!("  {d} {v:02x?}")));
                }
                if let Some((meta, data)) = r.doc_values_for_field(info.number) {
                    if let Some(e) = meta.binary_entry(info.number) {
                        let mut values = lucene_codecs::doc_values::BinaryReader::new(data, e);
                        for doc in 0..r.max_doc {
                            if let Some(v) = values.value(doc).unwrap() {
                                out.push(format!("  dv {doc} {v:02x?}"));
                            }
                        }
                    }
                }
            }
        }
        out
    };
    let (rust, java) = (dump(tmp.path()), dump(&root().join("index")));
    assert_eq!(rust.len(), java.len());
    for (r, j) in rust.iter().zip(&java) {
        assert_eq!(r, j);
    }
}

#[test]
fn triangles_encode_and_decode_as_lucene_does() {
    let mut n = 0;
    for line in read("triangles.tsv").lines() {
        let p: Vec<&str> = line.split('\t').collect();
        let v: Vec<&str> = p[0].split(',').collect();
        let i = |k: usize| v[k].parse::<i32>().unwrap();
        let b = |k: usize| v[k] == "true";
        let encoded =
            ShapeField::encode_triangle(i(1), i(0), b(2), i(4), i(3), b(5), i(7), i(6), b(8));
        if p[1] == "E" {
            let e = encoded.unwrap_err().to_string();
            assert!(e.ends_with(p[2]), "{line}: {e}");
            continue;
        }
        let bytes = encoded.unwrap();
        let hex: String = bytes.iter().map(|x| format!("{x:02x}")).collect();
        assert_eq!(hex, p[1], "{line}");
        let t = ShapeField::decode_triangle(&bytes);
        let got = format!(
            "{},{},{},{},{},{},{},{},{},{}",
            t.a_x, t.a_y, t.b_x, t.b_y, t.c_x, t.c_y, t.ab, t.bc, t.ca, t.kind
        );
        assert_eq!(got, p[2], "{line}");
        n += 1;
    }
    assert!(n > 1000, "{n}");
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn shape_doc_values_build_and_relate_as_lucene_does() {
    let mut shapes = 0;
    let mut relations = 0;
    let mut current: Option<(bool, Vec<u8>)> = None;
    for line in read("doc_values.tsv").lines() {
        let p: Vec<&str> = line.split('\t').collect();
        if p[0] == "dv" {
            let xy = p[1] == "xy";
            let (bytes, header, centroid, bbox) = if xy {
                let gs = parse_xy(p[3]).unwrap();
                let field = if p[2] == "geometry" {
                    xy_doc_value("f", &gs[0])
                } else {
                    let fs: Vec<ShapeTriangle> =
                        gs.iter().flat_map(|g| xy_fields("f", g)).collect();
                    XYShape::create_doc_value_field_from_fields("f", &fs).unwrap()
                };
                let v = field.doc_values().values();
                let c = field.centroid();
                let b = field.bounding_box();
                (
                    v.binary_value().to_vec(),
                    header_of(v),
                    format!("{},{}", jf(c.x()), jf(c.y())),
                    format!(
                        "{},{},{},{}",
                        jf(b.min_x),
                        jf(b.max_x),
                        jf(b.min_y),
                        jf(b.max_y)
                    ),
                )
            } else {
                let gs = parse_latlon(p[3]).unwrap();
                let field = if p[2] == "geometry" {
                    latlon_doc_value("f", &gs[0])
                } else {
                    let fs: Vec<ShapeTriangle> =
                        gs.iter().flat_map(|g| latlon_fields("f", g)).collect();
                    LatLonShape::create_doc_value_field_from_fields("f", &fs).unwrap()
                };
                let v = field.doc_values().values();
                let c = field.centroid();
                let b = field.bounding_box();
                (
                    v.binary_value().to_vec(),
                    header_of(v),
                    format!("{},{}", jd(c.lat()), jd(c.lon())),
                    format!(
                        "{},{},{},{}",
                        jd(b.min_lat),
                        jd(b.max_lat),
                        jd(b.min_lon),
                        jd(b.max_lon)
                    ),
                )
            };
            assert_eq!(hex(&bytes), p[4], "bytes of {}", p[3]);
            assert_eq!(header, p[5..9].join("\t"), "header of {}", p[3]);
            assert_eq!(centroid, p[9], "centroid of {}", p[3]);
            assert_eq!(bbox, p[10], "bounding box of {}", p[3]);
            // And read back from the bytes alone.
            if xy {
                let dv = XYShapeDocValues::new(Cow::Borrowed(&bytes)).unwrap();
                assert_eq!(header_of(dv.values()), p[5..9].join("\t"));
            } else {
                let dv = LatLonShapeDocValues::new(Cow::Borrowed(&bytes)).unwrap();
                assert_eq!(header_of(dv.values()), p[5..9].join("\t"));
            }
            current = Some((xy, bytes));
            shapes += 1;
        } else {
            let (xy, bytes) = current.as_ref().unwrap();
            let got = if *xy {
                let c = XYGeometry::create(&parse_xy(p[1]).unwrap()).unwrap();
                XYShape::create_xy_shape_doc_values(bytes)
                    .unwrap()
                    .values()
                    .relate(c.as_ref())
                    .unwrap()
            } else {
                let c = LatLonGeometry::create(&parse_latlon(p[1]).unwrap()).unwrap();
                LatLonShape::create_lat_lon_shape_doc_values(bytes)
                    .unwrap()
                    .values()
                    .relate(c.as_ref())
                    .unwrap()
            };
            assert_eq!(java_relation(got), p[2], "{line}");
            relations += 1;
        }
    }
    assert!(shapes >= 140 && relations >= 700, "{shapes} {relations}");
}

/// `ShapeAccess.header`.
fn header_of(v: &lucene_index::document::ShapeDocValues<'_>) -> String {
    format!(
        "{}\t{},{},{},{}\t{},{}\t{}",
        v.number_of_terms(),
        v.encoded_min_x(),
        v.encoded_max_x(),
        v.encoded_min_y(),
        v.encoded_max_y(),
        v.encoded_centroid_x(),
        v.encoded_centroid_y(),
        v.highest_dimension()
    )
}

fn jd(v: f64) -> String {
    lucene_util::geo::java_double_string(v)
}

fn jf(v: f32) -> String {
    lucene_util::geo::java_float_string(v)
}

fn java_relation(r: lucene_util::geo::Relation) -> &'static str {
    match r {
        lucene_util::geo::Relation::CellInsideQuery => "CELL_INSIDE_QUERY",
        lucene_util::geo::Relation::CellOutsideQuery => "CELL_OUTSIDE_QUERY",
        lucene_util::geo::Relation::CellCrossesQuery => "CELL_CROSSES_QUERY",
    }
}
