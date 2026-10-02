//! The geo point query benchmark pair (M9 T9.2), against
//! `benchmarks/micro/java/GeoPointsMicro.java`: boxes, distances, polygons,
//! distance sorts and `nearest` over the million-point index the Java side
//! builds (`GeoPointsMicro build <dir>`), replaying its `geo-queries-v2.tsv`:
//! and (T9.2 review) the distance feature query's top 10, the geometry
//! query under `WITHIN`/`DISJOINT`/`CONTAINS` and over lines and circles,
//! and the cartesian box/distance/polygon queries. Every hit of a filter
//! goes through a counting collector on both sides; each case prints a
//! `#check` digest the report compares before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::{
    lat_lon_doc_values_field, lat_lon_point, xy_point_field, QueryRelation,
};
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::{Circle, LatLonGeometry, Line, Point, Polygon, XYPolygon};

use super::measure;

/// FNV-1a over 64-bit words, identical to `GeoPointsMicro.Fnv`.
struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, x: i64) {
        self.0 ^= x as u64;
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

fn check(case: &str, d: &Fnv, n: u64) {
    println!("#check\t{case}\t{:016x}\t{n}", d.0);
}

/// `GeoPointsMicro.Count`: counts every hit, needs no score.
struct Count(i64);

impl ScoringCollector for Count {
    fn collect(&mut self, _doc_id: i32, _score: f32) {
        self.0 += 1;
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
}

fn count(leaves: &[OpenSegment<'_>], q: &dyn DocumentQuery) -> i64 {
    let mut c = Count(0);
    for leaf in leaves {
        q.score_leaf(leaf, 1.0, &mut c).unwrap();
    }
    c.0
}

/// `GeoPointsMicro.parsePts`: `a b;a b;...` as its two coordinate lists.
fn pts(spec: &str) -> (Vec<f64>, Vec<f64>) {
    spec.split(';')
        .map(|p| {
            let (x, y) = p.split_once(' ').unwrap();
            (x.parse::<f64>().unwrap(), y.parse::<f64>().unwrap())
        })
        .unzip()
}

fn relation(s: &str) -> QueryRelation {
    match s {
        "WITHIN" => QueryRelation::Within,
        "DISJOINT" => QueryRelation::Disjoint,
        "CONTAINS" => QueryRelation::Contains,
        "INTERSECTS" => QueryRelation::Intersects,
        other => panic!("{other}"),
    }
}

pub fn bench_geo_points(w: Duration, m: Duration, dir: &str) {
    let text = std::fs::read_to_string(format!("{dir}/geo-queries-v2.tsv"))
        .expect("run GeoPointsMicro build first (scripts/bench-micro.sh --bench geo_points)");
    let d = |s: &str| s.parse::<f64>().unwrap();
    let mut boxes = Vec::new();
    let mut dists = Vec::new();
    let mut polys = Vec::new();
    let mut sorts = Vec::new();
    let mut nearest = Vec::new();
    let mut features = Vec::new();
    let mut poly_within = Vec::new();
    let mut poly_disjoint = Vec::new();
    let mut pts_contains = Vec::new();
    let mut lines = Vec::new();
    let mut circle_within = Vec::new();
    let mut xy_boxes = Vec::new();
    let mut xy_dists = Vec::new();
    let mut xy_polys = Vec::new();
    let f32_of = |s: &str| s.parse::<f64>().unwrap() as f32;
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "box" => boxes.push(
                lat_lon_point::new_box_query("p", d(a[1]), d(a[2]), d(a[3]), d(a[4])).unwrap(),
            ),
            "dist" => dists
                .push(lat_lon_point::new_distance_query("p", d(a[1]), d(a[2]), d(a[3])).unwrap()),
            "poly" => {
                let (lats, lons): (Vec<f64>, Vec<f64>) = a[1]
                    .split(';')
                    .map(|p| {
                        let (x, y) = p.split_once(' ').unwrap();
                        (d(x), d(y))
                    })
                    .unzip();
                let p = Polygon::new(&lats, &lons, vec![]).unwrap();
                polys.push(lat_lon_point::new_polygon_query("p", &[p]).unwrap());
            }
            "sort" => sorts.push((d(a[1]), d(a[2]))),
            "nearest" => nearest.push((d(a[1]), d(a[2]))),
            "feature" => features.push(
                lat_lon_point::new_distance_feature_query("p", 1.0, d(a[1]), d(a[2]), d(a[3]))
                    .unwrap(),
            ),
            "gpoly" => {
                let (lats, lons) = pts(a[2]);
                let g = [LatLonGeometry::Polygon(
                    Polygon::new(&lats, &lons, vec![]).unwrap(),
                )];
                let q = lat_lon_point::new_geometry_query("p", relation(a[1]), &g).unwrap();
                if a[1] == "WITHIN" {
                    poly_within.push(q);
                } else {
                    poly_disjoint.push(q);
                }
            }
            "gpts" => {
                let (lats, lons) = pts(a[2]);
                let g: Vec<LatLonGeometry> = lats
                    .iter()
                    .zip(&lons)
                    .map(|(&la, &lo)| LatLonGeometry::Point(Point::new(la, lo).unwrap()))
                    .collect();
                pts_contains
                    .push(lat_lon_point::new_geometry_query("p", relation(a[1]), &g).unwrap());
            }
            "gline" => {
                let (lats, lons) = pts(a[2]);
                let g = [LatLonGeometry::Line(Line::new(&lats, &lons).unwrap())];
                lines.push(lat_lon_point::new_geometry_query("p", relation(a[1]), &g).unwrap());
            }
            "gcircle" => {
                let g = [LatLonGeometry::Circle(
                    Circle::new(d(a[2]), d(a[3]), d(a[4])).unwrap(),
                )];
                circle_within
                    .push(lat_lon_point::new_geometry_query("p", relation(a[1]), &g).unwrap());
            }
            "xybox" => xy_boxes.push(Box::new(
                xy_point_field::new_box_query(
                    "xy",
                    f32_of(a[1]),
                    f32_of(a[2]),
                    f32_of(a[3]),
                    f32_of(a[4]),
                )
                .unwrap(),
            ) as Box<dyn DocumentQuery>),
            "xydist" => xy_dists.push(Box::new(
                xy_point_field::new_distance_query("xy", f32_of(a[1]), f32_of(a[2]), f32_of(a[3]))
                    .unwrap(),
            ) as Box<dyn DocumentQuery>),
            "xypoly" => {
                let (x, y) = pts(a[1]);
                let x: Vec<f32> = x.into_iter().map(|v| v as f32).collect();
                let y: Vec<f32> = y.into_iter().map(|v| v as f32).collect();
                let p = XYPolygon::new(&x, &y, vec![]).unwrap();
                xy_polys.push(
                    Box::new(xy_point_field::new_polygon_query("xy", &[p]).unwrap())
                        as Box<dyn DocumentQuery>,
                );
            }
            other => panic!("{other}"),
        }
    }
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    for (name, qs) in [
        ("geo_box", &boxes),
        ("geo_distance", &dists),
        ("geo_polygon", &polys),
        ("geo_polygon_within", &poly_within),
        ("geo_polygon_disjoint", &poly_disjoint),
        ("geo_points_contains", &pts_contains),
        ("geo_line", &lines),
        ("geo_circle_within", &circle_within),
        ("geo_xy_box", &xy_boxes),
        ("geo_xy_distance", &xy_dists),
        ("geo_xy_polygon", &xy_polys),
    ] {
        let mut f = Fnv::new();
        for q in qs.iter() {
            f.add(count(&leaves, q.as_ref()));
        }
        check(name, &f, qs.len() as u64);
        measure(name, w, m, || {
            let mut n = 0;
            for q in black_box(qs.iter()) {
                n += count(&leaves, q.as_ref());
            }
            black_box(n);
            qs.len() as u64
        });
    }
    let mut f = Fnv::new();
    for q in &features {
        let td = dq::search_top_docs(&leaves, q.as_ref(), 10).unwrap();
        f.add(td.total_hits.value as i64);
        for h in &td.score_docs {
            f.add(i64::from(h.doc_id));
            f.add(i64::from(h.score.to_bits() as i32));
        }
    }
    check("geo_distance_feature", &f, features.len() as u64);
    measure("geo_distance_feature", w, m, || {
        for q in black_box(&features) {
            black_box(
                dq::search_top_docs(&leaves, q.as_ref(), 10)
                    .unwrap()
                    .score_docs[0]
                    .doc_id,
            );
        }
        features.len() as u64
    });
    let sort = |o: &(f64, f64)| {
        lat_lon_doc_values_field::new_distance_sort("p", o.0, o.1)
            .unwrap()
            .search(&leaves, &dq::MatchAllDocs, 10)
            .unwrap()
    };
    let mut f = Fnv::new();
    for o in &sorts {
        for (doc, v) in sort(o).hits {
            f.add(i64::from(doc));
            f.add(v.to_bits() as i64);
        }
    }
    check("geo_distance_sort", &f, sorts.len() as u64);
    measure("geo_distance_sort", w, m, || {
        for o in black_box(&sorts) {
            black_box(sort(o).hits[0].0);
        }
        sorts.len() as u64
    });
    let near = |o: &(f64, f64)| lat_lon_point::nearest(&leaves, "p", o.0, o.1, 10).unwrap();
    let mut f = Fnv::new();
    for o in &nearest {
        for (doc, v) in near(o).hits {
            f.add(i64::from(doc));
            f.add(v.to_bits() as i64);
        }
    }
    check("geo_nearest", &f, nearest.len() as u64);
    measure("geo_nearest", w, m, || {
        for o in black_box(&nearest) {
            black_box(near(o).hits[0].0);
        }
        nearest.len() as u64
    });
}
