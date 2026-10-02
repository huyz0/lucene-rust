//! The spatial3d query benchmark pair (M9 T9.4), against
//! `benchmarks/micro/java/Geo3dPointsMicro.java`: `PointInGeo3DShapeQuery`
//! over circles, boxes, polygons and paths, and the distance and
//! outside-distance sorts, over the 300 000-point WGS84 index the Java side
//! builds (`Geo3dPointsMicro build <dir>`), replaying its
//! `geo3d-queries.tsv`. Every hit of a filter goes through a counting
//! collector on both sides; each case prints a `#check` digest the report
//! compares before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::geo3d::{geo3d_doc_values_field, geo3d_point};
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::Polygon;
use lucene_util::spatial3d::PlanetModel;

use super::measure;

/// FNV-1a over 64-bit words, identical to `Geo3dPointsMicro.Fnv`.
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

/// `Geo3dPointsMicro.Count`: counts every hit, needs no score.
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

fn list(s: &str) -> Vec<f64> {
    s.split(';').map(|v| v.parse().unwrap()).collect()
}

/// Java's `Math.round(double)`: `floor(v + 0.5)`.
fn round(v: f64) -> i64 {
    (v + 0.5).floor() as i64
}

pub fn bench_geo3d_points(w: Duration, m: Duration, dir: &str) {
    let text = std::fs::read_to_string(format!("{dir}/geo3d-queries.tsv"))
        .expect("run Geo3dPointsMicro build first (scripts/bench-micro.sh --bench geo3d_points)");
    let pm = PlanetModel::wgs84();
    let d = |s: &str| s.parse::<f64>().unwrap();
    let (mut dists, mut boxes, mut polys, mut paths) = (vec![], vec![], vec![], vec![]);
    let (mut sorts, mut osorts) = (vec![], vec![]);
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "dist" => dists.push(
                geo3d_point::new_distance_query("p", &pm, d(a[1]), d(a[2]), d(a[3])).unwrap(),
            ),
            "box" => boxes.push(
                geo3d_point::new_box_query("p", &pm, d(a[1]), d(a[2]), d(a[3]), d(a[4])).unwrap(),
            ),
            "poly" => {
                let (lats, lons): (Vec<f64>, Vec<f64>) = a[1]
                    .split(';')
                    .map(|p| {
                        let (x, y) = p.split_once(' ').unwrap();
                        (d(x), d(y))
                    })
                    .unzip();
                let p = Polygon::new(&lats, &lons, vec![]).unwrap();
                polys.push(geo3d_point::new_polygon_query("p", &pm, &[p]).unwrap());
            }
            "path" => paths.push(
                geo3d_point::new_path_query("p", &list(a[1]), &list(a[2]), d(a[3]), &pm).unwrap(),
            ),
            "sort" => sorts.push(
                geo3d_doc_values_field::new_distance_sort("p", d(a[1]), d(a[2]), d(a[3]), &pm)
                    .unwrap(),
            ),
            "osort" => osorts.push(
                geo3d_doc_values_field::new_outside_distance_sort(
                    "p",
                    d(a[1]),
                    d(a[2]),
                    d(a[3]),
                    &pm,
                )
                .unwrap(),
            ),
            other => panic!("{other}"),
        }
    }
    let reader = DirectoryReader::open(&FsDirectory::open(std::path::Path::new(dir))).unwrap();
    let opened = reader.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    for (name, qs) in [
        ("geo3d_query_distance", &dists),
        ("geo3d_query_box", &boxes),
        ("geo3d_query_polygon", &polys),
        ("geo3d_query_path", &paths),
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
    for s in &sorts {
        for (doc, v) in s.search(&leaves, &dq::MatchAllDocs, 10).unwrap().hits {
            f.add(i64::from(doc));
            f.add(round(v));
        }
    }
    check("geo3d_distance_sort", &f, sorts.len() as u64);
    measure("geo3d_distance_sort", w, m, || {
        for s in black_box(&sorts) {
            black_box(s.search(&leaves, &dq::MatchAllDocs, 10).unwrap().hits[0].0);
        }
        sorts.len() as u64
    });
    let mut f = Fnv::new();
    for s in &osorts {
        for (doc, v) in s.search(&leaves, &dq::MatchAllDocs, 10).unwrap().hits {
            f.add(i64::from(doc));
            f.add(round(v));
        }
    }
    check("geo3d_outside_sort", &f, osorts.len() as u64);
    measure("geo3d_outside_sort", w, m, || {
        for s in black_box(&osorts) {
            black_box(s.search(&leaves, &dq::MatchAllDocs, 10).unwrap().hits[0].0);
        }
        osorts.len() as u64
    });
}
