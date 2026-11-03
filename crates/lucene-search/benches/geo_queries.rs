//! End-to-end cost of the geo point, shape, doc-values and Geo3D queries and
//! of the distance sort, over one synthetic index built in memory.
//!
//! The index holds 200,000 documents, 30% of them deleted:
//! - `p`: `LatLonPoint` + `LatLonDocValuesField`, uniform over
//!   `[-10, 10] x [-10, 10]` degrees;
//! - `sparse`: `LatLonDocValuesField` on 10% of the documents;
//! - `g`: `Geo3DPoint` (WGS84) at the same location;
//! - `xy`: `XYPointField` at the same coordinates;
//! - `s` / `xys`: a 0.05-degree `LatLonShape` / `XYShape` square on 20%.
//!
//! Each case times one full query (or top-n sort) over every leaf. The
//! shapes exercise the per-document decoders, the `ComponentTree` and edge
//! tree predicates, the Geo3D cell relation, the doc-values two-phase loop
//! and the distance comparator.
//!
//! Run with `cargo bench -p lucene-search --bench geo_queries`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_index::buffered_updates::Term;
use lucene_index::document::{
    Document, Geo3DPoint, LatLonDocValuesField, LatLonPoint, LatLonShape, Store, StringField,
    XYPointField, XYShape,
};
use lucene_index::index_writer::IndexWriter;
use lucene_index::merge_policy::upgrade::LATEST;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::geo3d::geo3d_point;
use lucene_search::document::geo::{
    lat_lon_doc_values_field, lat_lon_point, lat_lon_shape, xy_point_field, xy_shape, QueryRelation,
};
use lucene_search::document::{search_all, DocumentQuery, MatchAllDocs};
use lucene_store::ByteBuffersDirectory;
use lucene_util::geo::{LatLonGeometry, Polygon, XYPolygon};
use lucene_util::spatial3d::PlanetModel;

const DOCS: usize = 200_000;

/// SplitMix64, so every run indexes the same documents.
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    }
}

fn square(lat: f64, lon: f64, h: f64) -> Polygon {
    Polygon::new(
        &[lat, lat, lat + h, lat + h, lat],
        &[lon, lon + h, lon + h, lon, lon],
        vec![],
    )
    .unwrap()
}

fn xy_square(x: f32, y: f32, h: f32) -> XYPolygon {
    XYPolygon::new(&[x, x + h, x + h, x, x], &[y, y, y + h, y + h, y], vec![]).unwrap()
}

/// A jittered circle of `n` vertices, counter-clockwise.
fn ring(rng: &mut Rng, lat: f64, lon: f64, r: f64, n: usize) -> Polygon {
    let mut lats = Vec::with_capacity(n + 1);
    let mut lons = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = i as f64 / n as f64 * std::f64::consts::TAU;
        let rr = r * rng.f64_in(0.9, 1.0);
        lats.push(lat + rr * t.sin());
        lons.push(lon + rr * t.cos());
    }
    lats.push(lats[0]);
    lons.push(lons[0]);
    Polygon::new(&lats, &lons, vec![]).unwrap()
}

fn build() -> ByteBuffersDirectory {
    let dir = ByteBuffersDirectory::new();
    let mut rng = Rng(42);
    let mut w = IndexWriter::open(&dir, Vec::new(), "Lucene104", LATEST).unwrap();
    w.set_ram_buffer_size_mb(4096.0).unwrap();
    for _ in 0..DOCS {
        let lat = rng.f64_in(-10.0, 10.0);
        let lon = rng.f64_in(-10.0, 10.0);
        let mut d = Document::new();
        if rng.next_u64() % 10 < 3 {
            d.add(StringField::new("del", "y", Store::No));
        }
        d.add(LatLonPoint::new("p", lat, lon).unwrap());
        d.add(LatLonDocValuesField::new("p", lat, lon).unwrap());
        if rng.next_u64().is_multiple_of(10) {
            d.add(LatLonDocValuesField::new("sparse", lat, lon).unwrap());
        }
        d.add(Geo3DPoint::new("g", lat, lon).unwrap());
        d.add(XYPointField::new("xy", lon as f32, lat as f32).unwrap());
        if rng.next_u64().is_multiple_of(5) {
            for t in LatLonShape::create_indexable_fields("s", &square(lat, lon, 0.05)).unwrap() {
                d.add(t);
            }
            for t in
                XYShape::create_indexable_fields("xys", &xy_square(lon as f32, lat as f32, 0.05))
                    .unwrap()
            {
                d.add(t);
            }
        }
        w.add_fields_document(&d).unwrap();
    }
    w.delete_documents_by_term(&[Term::new("del", "y")])
        .unwrap();
    w.commit().unwrap();
    drop(w);
    dir
}

fn bench(c: &mut Criterion) {
    let dir = build();
    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let leaves = opened.as_open_segments();
    let mut rng = Rng(7);
    let poly1k = ring(&mut rng, 1.0, 1.0, 3.0, 1000);
    let multi: Vec<LatLonGeometry> = (0..200)
        .map(|i| {
            let lat = -9.0 + (i / 20) as f64 * 1.8;
            let lon = -9.5 + (i % 20) as f64 * 0.95;
            LatLonGeometry::Polygon(ring(&mut rng, lat, lon, 0.3, 20))
        })
        .collect();
    let xy_poly = {
        let p = ring(&mut rng, 1.0, 1.0, 3.0, 1000);
        let x: Vec<f32> = p.poly_lons().iter().map(|&v| v as f32).collect();
        let y: Vec<f32> = p.poly_lats().iter().map(|&v| v as f32).collect();
        XYPolygon::new(&x, &y, vec![]).unwrap()
    };
    let pm = PlanetModel::wgs84();

    let queries: Vec<(&str, Box<dyn DocumentQuery>)> = vec![
        (
            "ll_distance_50km",
            lat_lon_point::new_distance_query("p", 0.5, 0.5, 50_000.0).unwrap(),
        ),
        (
            "ll_distance_1000km",
            lat_lon_point::new_distance_query("p", 0.5, 0.5, 1_000_000.0).unwrap(),
        ),
        (
            "ll_polygon_1k",
            lat_lon_point::new_polygon_query("p", std::slice::from_ref(&poly1k)).unwrap(),
        ),
        (
            "ll_multipolygon_200",
            lat_lon_point::new_geometry_query("p", QueryRelation::Intersects, &multi).unwrap(),
        ),
        (
            "xy_polygon_1k",
            Box::new(xy_point_field::new_polygon_query("xy", &[xy_poly]).unwrap()),
        ),
        (
            "shape_box",
            lat_lon_shape::new_box_query("s", QueryRelation::Intersects, -2.0, 2.0, -2.0, 2.0)
                .unwrap(),
        ),
        (
            "shape_polygon_1k",
            lat_lon_shape::new_polygon_query(
                "s",
                QueryRelation::Intersects,
                std::slice::from_ref(&poly1k),
            )
            .unwrap(),
        ),
        (
            "shape_multipolygon_200",
            lat_lon_shape::new_geometry_query("s", QueryRelation::Intersects, &multi).unwrap(),
        ),
        (
            "xy_shape_box",
            xy_shape::new_box_query("xys", QueryRelation::Intersects, -2.0, 2.0, -2.0, 2.0)
                .unwrap(),
        ),
        (
            "geo3d_circle_100km",
            geo3d_point::new_distance_query("g", &pm, 0.5, 0.5, 100_000.0).unwrap(),
        ),
        (
            "geo3d_large_polygon_1k",
            geo3d_point::new_large_polygon_query("g", &pm, std::slice::from_ref(&poly1k)).unwrap(),
        ),
        (
            "dv_polygon_1k",
            lat_lon_doc_values_field::new_slow_polygon_query("p", std::slice::from_ref(&poly1k))
                .unwrap(),
        ),
        (
            "dv_sparse_polygon_1k",
            lat_lon_doc_values_field::new_slow_polygon_query(
                "sparse",
                std::slice::from_ref(&poly1k),
            )
            .unwrap(),
        ),
    ];

    let mut group = c.benchmark_group("geo_queries");
    group.sample_size(20);
    for (name, q) in &queries {
        group.bench_function(*name, |b| {
            b.iter(|| black_box(search_all(&leaves, q.as_ref()).unwrap().len()))
        });
    }
    let sort = lat_lon_doc_values_field::new_distance_sort("p", 0.5, 0.5).unwrap();
    for n in [10usize, 10_000] {
        group.bench_function(format!("distance_sort_top{n}"), |b| {
            b.iter(|| black_box(sort.search(&leaves, &MatchAllDocs, n).unwrap().hits.len()))
        });
    }
    for n in [10usize, 10_000] {
        group.bench_function(format!("nearest_{n}"), |b| {
            b.iter(|| {
                black_box(
                    lat_lon_point::nearest(&leaves, "p", 0.5, 0.5, n as i32)
                        .unwrap()
                        .hits
                        .len(),
                )
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
