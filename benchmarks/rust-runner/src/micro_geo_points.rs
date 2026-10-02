//! The geo point query benchmark pair (M9 T9.2), against
//! `benchmarks/micro/java/GeoPointsMicro.java`: boxes, distances, polygons,
//! distance sorts and `nearest` over the million-point index the Java side
//! builds (`GeoPointsMicro build <dir>`), replaying its `geo-queries.tsv`.
//! Every hit goes through a counting collector on both sides; each case
//! prints a `#check` digest the report compares before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::{lat_lon_doc_values_field, lat_lon_point};
use lucene_search::document::{self as dq, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::Polygon;

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

pub fn bench_geo_points(w: Duration, m: Duration, dir: &str) {
    let text = std::fs::read_to_string(format!("{dir}/geo-queries.tsv"))
        .expect("run GeoPointsMicro build first (scripts/bench-micro.sh --bench geo_points)");
    let d = |s: &str| s.parse::<f64>().unwrap();
    let mut boxes = Vec::new();
    let mut dists = Vec::new();
    let mut polys = Vec::new();
    let mut sorts = Vec::new();
    let mut nearest = Vec::new();
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "box" => boxes
                .push(lat_lon_point::new_box_query("p", d(a[1]), d(a[2]), d(a[3]), d(a[4])).unwrap()),
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
