//! spatial3d's field, query and sorts, differentially against Lucene
//! 10.5.0: `fixtures/src/GenGeo3dPoints.java` indexed a seeded corpus of
//! `Geo3DPoint`/`Geo3DDocValuesField` points (WGS84 and the sphere; poles,
//! the dateline, duplicates, clusters, multi-valued documents, documents
//! without the field, four segments with deletions; and a one-segment index
//! whose latitudes rise with the doc id) and recorded its answer to every
//! `Geo3DPoint` query factory and every `Geo3DDocValuesField` sort. This
//! builds the same queries and sorts and requires the same hits and
//! sort-value bits -- over Java's index, and over an index this port writes
//! from the same documents.

#![allow(clippy::arithmetic_side_effects)]

use std::sync::Arc;

use lucene_index::buffered_updates::Term;
use lucene_index::document::geo3d::from_degrees;
use lucene_index::document::{Document, Geo3DDocValuesField, Geo3DPoint, Store, StringField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::geo3d::{geo3d_doc_values_field as dvf, geo3d_point};
use lucene_search::document::geo::{
    Geo3DPointOutsideSortField, Geo3DPointSortField, SortedDistance,
};
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::Polygon;
use lucene_util::spatial3d::{GeoPoint, PlanetModel};
use lucene_util::test_support::TempDir;

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo3d_points")
}

fn read(file: &str) -> String {
    std::fs::read_to_string(root().join(file))
        .expect("run scripts/gen-fixtures.sh --only GenGeo3dPoints")
}

fn model(field: &str) -> Arc<PlanetModel> {
    if field == "s" {
        PlanetModel::sphere()
    } else {
        PlanetModel::wgs84()
    }
}

fn d(s: &str) -> f64 {
    s.parse().unwrap()
}

fn list(s: &str) -> Vec<f64> {
    s.split(';').map(d).collect()
}

fn ring(s: &str) -> (Vec<f64>, Vec<f64>) {
    s.split(';')
        .map(|p| {
            let (x, y) = p.split_once(' ').unwrap();
            (d(x), d(y))
        })
        .unzip()
}

/// `GeoCorpus.spec` polygons (`G:` rings joined by ` + `).
fn polygons(spec: &str) -> Vec<Polygon> {
    spec.split(" + ")
        .map(|g| {
            let body = g.strip_prefix("G:").unwrap();
            let mut rings = body.split('|');
            let (lats, lons) = ring(rings.next().unwrap());
            let holes = rings
                .map(|r| {
                    let (la, lo) = ring(r);
                    Polygon::new(&la, &lo, vec![]).unwrap()
                })
                .collect();
            Polygon::new(&lats, &lons, holes).unwrap()
        })
        .collect()
}

fn hex_bits(hits: &[i32]) -> String {
    let max = hits.iter().copied().max().unwrap_or(-1);
    let mut b = vec![0u8; ((max + 8) / 8) as usize];
    for &doc in hits {
        b[(doc >> 3) as usize] |= 1 << (doc & 7);
    }
    while b.last() == Some(&0) {
        b.pop();
    }
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A failure as the generator writes it: Java's class (all
/// `IllegalArgumentException`s here) and message.
fn err(e: &lucene_search::Error) -> String {
    let msg = e.to_string();
    let msg = msg.split_once(": ").map_or(&*msg, |m| m.1);
    // Java's NullPointerException (an unset bound) has the JVM's message;
    // it is compared by class.
    if msg.contains("(NullPointerException in Java)") {
        return "E\tjava.lang.NullPointerException".into();
    }
    format!(
        "E\tjava.lang.IllegalArgumentException\t{}",
        msg.replace('\\', "\\\\")
            .replace('\t', "\\t")
            .replace('\n', "\\n")
    )
}

/// An answer as recorded, with a `NullPointerException`'s message dropped.
fn normalized(want: &str) -> String {
    if want.starts_with("E\tjava.lang.NullPointerException") {
        return "E\tjava.lang.NullPointerException".into();
    }
    want.to_string()
}

fn constant(
    segments: &[OpenSegment<'_>],
    q: lucene_search::Result<Box<dyn DocumentQuery>>,
) -> String {
    let q = match q {
        Ok(q) => q,
        Err(e) => return err(&e),
    };
    let hits = match dq::search_all(segments, q.as_ref()) {
        Ok(h) => h,
        Err(e) => return err(&e),
    };
    assert!(hits.iter().all(|h| h.score == 1.0), "not constant");
    let ids: Vec<i32> = hits.iter().map(|h| h.doc_id).collect();
    format!("C\t{}\t{}", ids.len(), hex_bits(&ids))
}

fn sorted(r: lucene_search::Result<SortedDistance>) -> String {
    match r {
        Err(e) => err(&e),
        Ok(r) => {
            let hits: Vec<String> = r
                .hits
                .iter()
                .map(|(doc, v)| format!("{doc}:{:x}", v.to_bits()))
                .collect();
            format!("T\t{}\t{}", r.total_hits, hits.join(","))
        }
    }
}

enum Sort {
    Inside(Geo3DPointSortField),
    Outside(Geo3DPointOutsideSortField),
}

/// The sort a `queries.tsv` sort spec names (`a` from its kind on).
fn sort_field(field: &str, pm: &Arc<PlanetModel>, a: &[&str]) -> lucene_search::Result<Sort> {
    Ok(match a[0] {
        "dist" => Sort::Inside(dvf::new_distance_sort(
            field,
            d(a[1]),
            d(a[2]),
            d(a[3]),
            pm,
        )?),
        "path" => Sort::Inside(dvf::new_path_sort(
            field,
            &list(a[1]),
            &list(a[2]),
            d(a[3]),
            pm,
        )?),
        "odist" => Sort::Outside(dvf::new_outside_distance_sort(
            field,
            d(a[1]),
            d(a[2]),
            d(a[3]),
            pm,
        )?),
        "obox" => Sort::Outside(dvf::new_outside_box_sort(
            field,
            d(a[1]),
            d(a[2]),
            d(a[3]),
            d(a[4]),
            pm,
        )?),
        "opoly" => Sort::Outside(dvf::new_outside_polygon_sort(field, pm, &polygons(a[1]))?),
        "olpoly" => Sort::Outside(dvf::new_outside_large_polygon_sort(
            field,
            pm,
            &polygons(a[1]),
        )?),
        "opath" => Sort::Outside(dvf::new_outside_path_sort(
            field,
            &list(a[1]),
            &list(a[2]),
            d(a[3]),
            pm,
        )?),
        other => panic!("unknown sort {other}"),
    })
}

/// Runs one `queries.tsv` query, formatted as `GenGeo3dPoints` formats it.
fn run(segments: &[OpenSegment<'_>], a: &[&str]) -> String {
    let field = a[1];
    let pm = model(field);
    match a[0] {
        "dist" => constant(
            segments,
            geo3d_point::new_distance_query(field, &pm, d(a[2]), d(a[3]), d(a[4])),
        ),
        "box" => constant(
            segments,
            geo3d_point::new_box_query(field, &pm, d(a[2]), d(a[3]), d(a[4]), d(a[5])),
        ),
        "poly" => constant(
            segments,
            geo3d_point::new_polygon_query(field, &pm, &polygons(a[2])),
        ),
        "lpoly" => constant(
            segments,
            geo3d_point::new_large_polygon_query(field, &pm, &polygons(a[2])),
        ),
        "path" => constant(
            segments,
            geo3d_point::new_path_query(field, &list(a[2]), &list(a[3]), d(a[4]), &pm),
        ),
        "sort" => {
            let n: usize = a[2].parse().unwrap();
            let base: Box<dyn DocumentQuery> = if a[3] == "all" {
                Box::new(dq::MatchAllDocs)
            } else {
                let v: Vec<f64> = a[3]
                    .strip_prefix("box:")
                    .unwrap()
                    .split(',')
                    .map(d)
                    .collect();
                let points = if field == "pd" { "p" } else { field };
                geo3d_point::new_box_query(points, &pm, v[0], v[1], v[2], v[3]).unwrap()
            };
            match sort_field(field, &pm, &a[4..]) {
                Err(e) => err(&e),
                Ok(Sort::Inside(s)) => sorted(s.search(segments, base.as_ref(), n)),
                Ok(Sort::Outside(s)) => sorted(s.search(segments, base.as_ref(), n)),
            }
        }
        other => panic!("unknown query {other}"),
    }
}

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
        let want = normalized(want);
        if got != want {
            let short = |s: &str| s.chars().take(300).collect::<String>();
            failures.push(format!(
                "{}\n  java: {}\n  rust: {}",
                short(query),
                short(&want),
                short(&got)
            ));
        }
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} queries differ:\n{}",
        failures.len(),
        failures[..failures.len().min(20)].join("\n")
    );
    n
}

/// `GenGeo3dPoints.addFields`: `p` a point, `pd` a doc value, `s` both
/// (on the sphere).
fn add_point(doc: &mut Document, field: &str, lat: f64, lon: f64) {
    let pm = model(field);
    if field != "pd" {
        doc.add(Geo3DPoint::with_planet_model(field, &pm, lat, lon).unwrap());
    }
    if field != "p" {
        add_doc_value(doc, field, &pm, lat, lon);
    }
}

fn add_doc_value(doc: &mut Document, field: &str, pm: &Arc<PlanetModel>, lat: f64, lon: f64) {
    let g = GeoPoint::from_lat_lon(pm, from_degrees(lat), from_degrees(lon)).unwrap();
    doc.add(Geo3DDocValuesField::new(field, &g, pm).unwrap());
}

fn with_writer(dir: &std::path::Path, f: impl FnOnce(&mut IndexWriter<'_>)) {
    let fs = FsDirectory::open(dir);
    let version = LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    };
    let mut w = IndexWriter::open(&fs, Vec::new(), "Lucene104", version).unwrap();
    f(&mut w);
}

/// The same documents, written by this port: every commit and delete
/// `GenGeo3dPoints.main` made, in order.
fn write_rust_index(dir: &std::path::Path) {
    with_writer(dir, write_docs);
}

fn write_docs(w: &mut IndexWriter<'_>) {
    for (i, line) in read("docs.tsv").lines().enumerate() {
        let mut p = line.split('\t');
        let id = p.next().unwrap();
        let mut doc = Document::new();
        doc.add(StringField::new("id", id, Store::Yes));
        for spec in p {
            let (field, ab) = spec.split_once(':').unwrap();
            let (a, b) = ab.split_once(',').unwrap();
            add_point(&mut doc, field, d(a), d(b));
        }
        w.add_fields_document(&doc).unwrap();
        if i + 1 == 1500 || i + 1 == 3000 || i + 1 == 4500 || i + 1 == 4550 {
            w.commit().unwrap();
        }
    }
    for id in read("deletes.tsv").lines() {
        w.delete_documents_by_term(&[Term::new("id", id)]).unwrap();
    }
    w.commit().unwrap();
}

fn write_big_index(dir: &std::path::Path) {
    with_writer(dir, write_big);
}

fn write_big(w: &mut IndexWriter<'_>) {
    for line in read("big/big.tsv").lines() {
        let (a, b) = line.split_once(',').unwrap();
        let mut doc = Document::new();
        add_point(&mut doc, "p", d(a), d(b));
        add_doc_value(&mut doc, "p", &PlanetModel::wgs84(), d(a), d(b));
        w.add_fields_document(&doc).unwrap();
    }
    w.commit().unwrap();
}

#[test]
fn every_geo3d_query_and_sort_matches_lucene_on_lucenes_index() {
    let n = check_query_file(&root().join("index"), "queries.tsv", 4);
    assert!(n >= 250, "{n} queries");
}

#[test]
fn every_geo3d_query_and_sort_matches_lucene_on_this_ports_index() {
    let tmp = TempDir::new("geo3d-points-write");
    write_rust_index(tmp.path());
    let n = check_query_file(tmp.path(), "queries.tsv", 4);
    assert!(n >= 250, "{n} queries");
}

#[test]
fn every_geo3d_point_is_indexed_as_lucene_indexes_it() {
    let tmp = TempDir::new("geo3d-points-bytes");
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
}

#[test]
fn the_big_segment_matches_lucene_on_both_indexes() {
    let n = check_query_file(&root().join("big/index"), "big/queries.tsv", 1);
    assert!(n >= 250, "{n} queries");
    let tmp = TempDir::new("geo3d-points-big");
    write_big_index(tmp.path());
    assert_eq!(
        check_query_file(tmp.path(), "big/queries.tsv", 1),
        n,
        "over this port's index"
    );
}
