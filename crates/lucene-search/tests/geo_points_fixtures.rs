//! The geo point fields, queries, sorts and `nearest`, differentially
//! against Lucene 10.5.0: `fixtures/src/GenGeoPoints.java` indexed a seeded
//! corpus of `LatLonPoint`/`LatLonDocValuesField`/`XYPointField`/
//! `XYDocValuesField` points (poles, the dateline, duplicates, clusters,
//! multi-valued documents, documents without the field, four segments with
//! deletions) and recorded its answer to ~1100 queries of every kind. This
//! builds the same queries with `lucene_search::document::geo` and requires
//! the same hits, score bits, sort-value bits and distances -- over Java's
//! index, and over an index this port writes from the same documents.

#![allow(clippy::arithmetic_side_effects)]

use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, LatLonDocValuesField, LatLonPoint, Store, StringField, XYDocValuesField, XYPointField,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::{
    self, lat_lon_doc_values_field, lat_lon_point, xy_doc_values_field, xy_point_field,
    QueryRelation,
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
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo_points")
}

fn read(file: &str) -> String {
    std::fs::read_to_string(root().join(file))
        .expect("run scripts/gen-fixtures.sh --only GenGeoPoints")
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

fn polygons(spec: &str) -> Vec<Polygon> {
    parse_latlon(spec)
        .unwrap()
        .into_iter()
        .map(|g| match g {
            LatLonGeometry::Polygon(p) => p,
            other => panic!("{other:?}"),
        })
        .collect()
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

fn constant(
    segments: &[OpenSegment<'_>],
    q: Result<Box<dyn DocumentQuery>, lucene_search::Error>,
) -> String {
    let q = match q {
        Ok(q) => q,
        Err(e) => {
            let msg = e.to_string();
            return format!("E\t{}", msg.split_once(": ").map_or(&*msg, |m| m.1));
        }
    };
    let hits = dq::search_all(segments, q.as_ref()).unwrap();
    assert!(hits.iter().all(|h| h.score == 1.0), "not constant");
    let ids: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
    format!("C\t{}\t{}", ids.len(), hex_bits(&ids))
}

fn bx<Q: DocumentQuery + 'static>(
    q: Result<Q, lucene_search::Error>,
) -> Result<Box<dyn DocumentQuery>, lucene_search::Error> {
    q.map(|q| Box::new(q) as Box<dyn DocumentQuery>)
}

fn filter(spec: &str) -> Box<dyn DocumentQuery> {
    if spec == "all" {
        return Box::new(dq::MatchAllDocs);
    }
    let v = nums(spec.strip_prefix("box:").unwrap());
    lat_lon_point::new_box_query("ll", v[0], v[1], v[2], v[3]).unwrap()
}

fn d(s: &str) -> f64 {
    s.parse().unwrap()
}

fn f(s: &str) -> f32 {
    s.parse().unwrap()
}

/// Runs one `queries.tsv` query, formatted as `GenGeoPoints` formats it.
fn run(segments: &[OpenSegment<'_>], a: &[&str]) -> String {
    let field = a[1];
    match a[0] {
        "box" => constant(
            segments,
            lat_lon_point::new_box_query(field, d(a[2]), d(a[3]), d(a[4]), d(a[5])),
        ),
        "dvbox" => constant(
            segments,
            lat_lon_doc_values_field::new_slow_box_query(field, d(a[2]), d(a[3]), d(a[4]), d(a[5])),
        ),
        "dist" => constant(
            segments,
            lat_lon_point::new_distance_query(field, d(a[2]), d(a[3]), d(a[4])),
        ),
        "dvdist" => constant(
            segments,
            lat_lon_doc_values_field::new_slow_distance_query(field, d(a[2]), d(a[3]), d(a[4])),
        ),
        "poly" => constant(
            segments,
            lat_lon_point::new_polygon_query(field, &polygons(a[2])),
        ),
        "dvpoly" => constant(
            segments,
            lat_lon_doc_values_field::new_slow_polygon_query(field, &polygons(a[2])),
        ),
        "geom" => constant(
            segments,
            lat_lon_point::new_geometry_query(field, relation(a[2]), &parse_latlon(a[3]).unwrap()),
        ),
        "dvgeom" => constant(
            segments,
            lat_lon_doc_values_field::new_slow_geometry_query(
                field,
                relation(a[2]),
                &parse_latlon(a[3]).unwrap(),
            ),
        ),
        "xygeom" => constant(
            segments,
            bx(xy_point_field::new_geometry_query(
                field,
                &parse_xy(a[2]).unwrap(),
            )),
        ),
        "xydvgeom" => constant(
            segments,
            bx(xy_doc_values_field::new_slow_geometry_query(
                field,
                &parse_xy(a[2]).unwrap(),
            )),
        ),
        "xybox" => constant(
            segments,
            bx(xy_point_field::new_box_query(
                field,
                f(a[2]),
                f(a[3]),
                f(a[4]),
                f(a[5]),
            )),
        ),
        "xydvbox" => constant(
            segments,
            bx(xy_doc_values_field::new_slow_box_query(
                field,
                f(a[2]),
                f(a[3]),
                f(a[4]),
                f(a[5]),
            )),
        ),
        "xydist" => constant(
            segments,
            bx(xy_point_field::new_distance_query(
                field,
                f(a[2]),
                f(a[3]),
                f(a[4]),
            )),
        ),
        "xydvdist" => constant(
            segments,
            bx(xy_doc_values_field::new_slow_distance_query(
                field,
                f(a[2]),
                f(a[3]),
                f(a[4]),
            )),
        ),
        "feature" => {
            let q = lat_lon_point::new_distance_feature_query(
                field,
                f(a[2]),
                d(a[3]),
                d(a[4]),
                d(a[5]),
            )
            .unwrap();
            let n: usize = a[6].parse().unwrap();
            let td = dq::search_top_docs(segments, q.as_ref(), n).unwrap();
            let rel = match td.total_hits.relation {
                TotalHitsRelation::EqualTo => "EQUAL_TO",
                TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
            };
            let hits: Vec<String> = td
                .score_docs
                .iter()
                .map(|s| format!("{}:{:x}", s.doc_id, s.score.to_bits()))
                .collect();
            format!("S\t{}\t{rel}\t{}", td.total_hits.value, hits.join(","))
        }
        "sort" | "xysort" => {
            let n: usize = a[4].parse().unwrap();
            let q = filter(a[5]);
            let r = if a[0] == "sort" {
                lat_lon_doc_values_field::new_distance_sort(field, d(a[2]), d(a[3]))
                    .unwrap()
                    .search(segments, q.as_ref(), n)
                    .unwrap()
            } else {
                xy_doc_values_field::new_distance_sort(field, f(a[2]), f(a[3]))
                    .search(segments, q.as_ref(), n)
                    .unwrap()
            };
            let hits: Vec<String> = r
                .hits
                .iter()
                .map(|(doc, v)| format!("{doc}:{:x}", v.to_bits()))
                .collect();
            format!("T\t{}\t{}", r.total_hits, hits.join(","))
        }
        "nearest" => {
            let n: i32 = a[4].parse().unwrap();
            let r = lat_lon_point::nearest(segments, field, d(a[2]), d(a[3]), n).unwrap();
            let hits: Vec<String> = r
                .hits
                .iter()
                .map(|(doc, v)| format!("{doc}:{:x}", v.to_bits()))
                .collect();
            format!("N\t{}\t{}", r.total_hits, hits.join(","))
        }
        other => panic!("unknown query {other}"),
    }
}

/// Every query against `dir`; returns how many ran.
fn check_queries(dir: &std::path::Path) -> usize {
    check_query_file(dir, "queries.tsv", 4)
}

/// Every query of `file` against `dir`, which must have `segments`
/// segments; returns how many ran.
fn check_query_file(dir: &std::path::Path, file: &str, segments_expected: usize) -> usize {
    let reader = DirectoryReader::open(&FsDirectory::open(dir)).expect("open");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), segments_expected, "segments");
    let mut failures = Vec::new();
    let mut n = 0;
    for line in read(file).lines() {
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
fn every_geo_query_matches_lucene_on_lucenes_index() {
    let n = check_queries(&root().join("index"));
    assert!(n > 1000, "{n} queries");
}

/// The same documents, written by this port: every commit and delete
/// `GenGeoPoints.main` made, in order.
fn write_rust_index(dir: &std::path::Path) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    let docs = read("docs.tsv");
    for (i, line) in docs.lines().enumerate() {
        let mut p = line.split('\t');
        let id = p.next().unwrap();
        let mut doc = Document::new();
        doc.add(StringField::new("id", id, Store::Yes));
        for spec in p {
            let (field, ab) = spec.split_once(':').unwrap();
            let (a, b) = ab.split_once(',').unwrap();
            if field == "xy" {
                let (x, y) = (f(a), f(b));
                doc.add(XYPointField::new("xy", x, y).unwrap());
                doc.add(XYDocValuesField::new("xy", x, y).unwrap());
            } else {
                let (lat, lon) = (d(a), d(b));
                doc.add(LatLonPoint::new(field, lat, lon).unwrap());
                doc.add(LatLonDocValuesField::new(field, lat, lon).unwrap());
            }
        }
        w.add_fields_document(&doc).unwrap();
        // GenGeoPoints commits after each 2000-document segment and the
        // 60-document one.
        if i + 1 == 2000 || i + 1 == 4000 || i + 1 == 6000 || i + 1 == 6060 {
            w.commit().unwrap();
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)]).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn every_geo_query_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("geo-points-write");
    write_rust_index(tmp.path());
    let n = check_queries(tmp.path());
    assert!(n > 1000, "{n} queries");
}

#[test]
fn every_point_is_indexed_as_lucene_indexes_it() {
    // The packed points and doc values of the two indexes, segment by
    // segment: the writer must encode exactly what Lucene encoded.
    let tmp = TempDir::new("geo-points-bytes");
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
                        "  {} points, {} docs, {:?}..{:?}: {:x}",
                        pf.point_count,
                        pf.doc_count,
                        pf.min_packed_value,
                        pf.max_packed_value,
                        all.iter().fold(0u64, |h, (d, v)| {
                            v.iter().fold(h ^ (*d as u64), |h, b| {
                                (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
                            })
                        })
                    ));
                }
            }
        }
        out
    };
    assert_eq!(dump(tmp.path()), dump(&root().join("index")));
    // And the doc values, through the doc-values queries: a whole-globe box
    // matches exactly the documents with a value.
    let reader = DirectoryReader::open(&FsDirectory::open(tmp.path())).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let all =
        lat_lon_doc_values_field::new_slow_box_query("one", -90.0, 90.0, -180.0, 180.0).unwrap();
    let hits = dq::search_all(&segments, all.as_ref()).unwrap();
    let live = 6000 - read("deletes.tsv").lines().count();
    assert_eq!(hits.len(), live);
    let _ = geo::QueryRelation::Within;
}

#[test]
fn distance_sorts_as_custom_keys_order_as_lucene_orders() {
    use lucene_search::query::MatchAllDocsQuery;
    use lucene_search::top_field::{register_comparator_source, search_sorted, SortField};
    use lucene_search::{BooleanQuery, Clause};
    let dir = root().join("index");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).expect("open");
    let mut opened = reader.open_segments().expect("open segments");
    opened.open_points().expect("open points");
    let segments = opened.as_open_segments();
    let norms: Vec<
        Option<&std::collections::HashMap<String, lucene_search::field_norms::FieldNorms<'_>>>,
    > = segments.iter().map(|_| None).collect();
    let match_all = BooleanQuery {
        must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(i32::MAX))],
        ..BooleanQuery::default()
    };
    let mut checked = 0;
    for line in read("queries.tsv").lines() {
        let (query, want) = line.split_once("\t=>\t").unwrap();
        let a: Vec<&str> = query.split('\t').collect();
        if !(a[0] == "sort" || a[0] == "xysort") || a[5] != "all" {
            continue;
        }
        let n: usize = a[4].parse().unwrap();
        let (source, latlon) = if a[0] == "sort" {
            let s = lat_lon_doc_values_field::new_distance_sort(a[1], d(a[2]), d(a[3])).unwrap();
            (s.comparator_source(), true)
        } else {
            let s = xy_doc_values_field::new_distance_sort(a[1], f(a[2]), f(a[3]));
            (s.comparator_source(), false)
        };
        let id = register_comparator_source(source);
        let td = search_sorted(
            &segments,
            reader.segment_readers(),
            &match_all,
            &norms,
            &[SortField::custom(a[1], id, false)],
            n,
            u64::MAX,
            None,
        )
        .unwrap();
        let hits: Vec<String> = td
            .hits
            .iter()
            .map(|h| {
                let key = sortable_long_to_double(h.values[0]);
                let v = if latlon && key.is_finite() {
                    lucene_util::sloppy_math::haversin_meters_from_sort_key(key)
                } else {
                    key
                };
                format!("{}:{:x}", h.doc, v.to_bits())
            })
            .collect();
        let got = format!("T\t{}\t{}", td.total.value, hits.join(","));
        assert_eq!(got, want, "{query}");
        checked += 1;
    }
    assert!(checked >= 20, "{checked}");
}

fn sortable_long_to_double(v: i64) -> f64 {
    lucene_index::document::sortable_long_to_double(v)
}

/// `geo_points/big`: one segment of 24 000 points, latitude rising with the
/// doc id -- the distance feature query's iterator narrowing, the sort
/// comparator's sampled bounding-box updates, a deep tree for `nearest`, and
/// the inverse walks of a fully populated single-valued segment.
fn write_big_index(dir: &std::path::Path) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    for line in read("big/big.tsv").lines() {
        let (a, b) = line.split_once(',').unwrap();
        let mut doc = Document::new();
        doc.add(LatLonPoint::new("p", d(a), d(b)).unwrap());
        doc.add(LatLonDocValuesField::new("p", d(a), d(b)).unwrap());
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn the_big_segment_matches_lucene_on_both_indexes() {
    let n = check_query_file(&root().join("big/index"), "big/queries.tsv", 1);
    assert!(n >= 80, "{n} queries");
    let tmp = TempDir::new("geo-points-big");
    write_big_index(tmp.path());
    assert_eq!(
        check_query_file(tmp.path(), "big/queries.tsv", 1),
        n,
        "over this port's index"
    );
}
