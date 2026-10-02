//! The shape benchmark pair (M9 T9.3), against
//! `benchmarks/micro/java/GeoShapesMicro.java`: tessellating and encoding
//! polygons into triangle fields and into shape doc values, and polygon
//! INTERSECTS / WITHIN, point CONTAINS and doc-values box queries over the
//! 200 000-shape index the Java side builds (`GeoShapesMicro build <dir>`).
//! Every hit goes through a counting collector on both sides; each case
//! prints a `#check` digest the report compares before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_index::document::{IndexableField, LatLonShape};
use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::{lat_lon_shape, QueryRelation};
use lucene_search::document::DocumentQuery;
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;
use lucene_util::geo::Polygon;

use super::measure;

/// FNV-1a over 64-bit words, identical to `GeoShapesMicro.Fnv`.
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

/// `GeoShapesMicro.Count`: counts every hit, needs no score.
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

fn d(s: &str) -> f64 {
    s.parse().unwrap()
}

/// `GeoShapesMicro.polygon`: `ring|hole|...`, each ring `lat lon;...`.
fn polygon(spec: &str) -> Polygon {
    let ring = |r: &str| -> (Vec<f64>, Vec<f64>) {
        r.split(';')
            .map(|p| {
                let (a, b) = p.split_once(' ').unwrap();
                (d(a), d(b))
            })
            .unzip()
    };
    let mut rings = spec.split('|');
    let (lats, lons) = ring(rings.next().unwrap());
    let holes = rings
        .map(|h| {
            let (a, b) = ring(h);
            Polygon::new(&a, &b, vec![]).unwrap()
        })
        .collect();
    Polygon::new(&lats, &lons, holes).unwrap()
}

/// Every byte as a long, as `GeoShapesMicro` digests them (`f.add(byte)`,
/// sign-extended).
fn digest_bytes(f: &mut Fnv, b: &[u8]) {
    for &x in b {
        f.add(i64::from(x as i8));
    }
}

pub fn bench_geo_shapes(w: Duration, m: Duration, dir: &str) {
    let polys: Vec<Polygon> = std::fs::read_to_string(format!("{dir}/geo-shapes.tsv"))
        .expect("run GeoShapesMicro build first (scripts/bench-micro.sh --bench geo_shapes)")
        .lines()
        .map(polygon)
        .collect();
    let mut f = Fnv::new();
    for p in &polys {
        for t in LatLonShape::create_indexable_fields("s", p).unwrap() {
            digest_bytes(&mut f, t.packed());
        }
    }
    check("shape_index_fields", &f, polys.len() as u64);
    measure("shape_index_fields", w, m, || {
        let mut n = 0;
        for p in black_box(&polys) {
            n += LatLonShape::create_indexable_fields("s", p).unwrap().len();
        }
        black_box(n);
        polys.len() as u64
    });
    let mut f = Fnv::new();
    for p in &polys {
        let dv = LatLonShape::create_doc_value_field("s", p).unwrap();
        digest_bytes(&mut f, &dv.binary_value().unwrap());
    }
    check("shape_index_doc_value", &f, polys.len() as u64);
    measure("shape_index_doc_value", w, m, || {
        let mut n = 0;
        for p in black_box(&polys) {
            let dv = LatLonShape::create_doc_value_field("s", p).unwrap();
            n += dv.binary_value().unwrap().len();
        }
        black_box(n);
        polys.len() as u64
    });

    let text = std::fs::read_to_string(format!("{dir}/geo-shape-queries.tsv")).unwrap();
    let mut intersects = Vec::new();
    let mut within = Vec::new();
    let mut contains = Vec::new();
    let mut dvbox = Vec::new();
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "intersects" => intersects.push(
                lat_lon_shape::new_polygon_query("s", QueryRelation::Intersects, &[polygon(a[1])])
                    .unwrap(),
            ),
            "within" => within.push(
                lat_lon_shape::new_polygon_query("s", QueryRelation::Within, &[polygon(a[1])])
                    .unwrap(),
            ),
            "contains" => contains.push(
                lat_lon_shape::new_point_query("s", QueryRelation::Contains, &[[d(a[1]), d(a[2])]])
                    .unwrap(),
            ),
            "dvbox" => dvbox.push(
                lat_lon_shape::new_slow_doc_values_box_query(
                    "s",
                    QueryRelation::Intersects,
                    d(a[1]),
                    d(a[2]),
                    d(a[3]),
                    d(a[4]),
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
        ("shape_intersects_polygon", &intersects),
        ("shape_within", &within),
        ("shape_contains_point", &contains),
        ("shape_doc_values_box", &dvbox),
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
}
