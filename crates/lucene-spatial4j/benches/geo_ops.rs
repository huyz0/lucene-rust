//! Microbenchmarks for the per-document geo paths in `lucene-util`:
//! `Polygon2D` point-in-polygon and box relations (the edge tree), the
//! grid predicates the point/doc-values queries test every candidate with,
//! the lat/lon quantization, `ComponentTree` over many polygons, the GeoJSON
//! and WKT parsers, the tessellator, and the spatial-extras prefix-tree
//! iterators and cell comparisons.
//!
//! Inputs are generated from a fixed seed so runs are comparable.
//!
//! Run with: `cargo bench -p lucene-util --bench geo_ops`

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_spatial4j::spatial4j::{Shape, SpatialContext, SpatialContextFactory};
use lucene_spatial4j::spatial_extras::prefix_tree::{
    Calendar, Cell, DateRangePrefixTree, GeohashPrefixTree, QuadPrefixTree, S2PrefixTree,
    SpatialPrefixTree,
};
use lucene_util::geo::{
    simple_wkt_shape_parser, tessellator, GeoEncodingUtils, GeoUtils, LatLonGeometry, Polygon,
    Rectangle, XYGeometry, XYRectangle,
};

/// SplitMix64: a tiny deterministic generator (no external dependency).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    fn f64_in(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64 * (hi - lo)
    }
}

/// A closed ring of `n` vertices on a jittered circle (counter-clockwise
/// when `ccw`).
fn ring(rng: &mut Rng, lat: f64, lon: f64, r: f64, n: usize, ccw: bool) -> (Vec<f64>, Vec<f64>) {
    let mut lats = Vec::with_capacity(n + 1);
    let mut lons = Vec::with_capacity(n + 1);
    for i in 0..n {
        let t = i as f64 / n as f64 * std::f64::consts::TAU;
        let t = if ccw { t } else { -t };
        let rr = r * rng.f64_in(0.9, 1.0);
        lats.push(lat + rr * t.sin());
        lons.push(lon + rr * t.cos());
    }
    lats.push(lats[0]);
    lons.push(lons[0]);
    (lats, lons)
}

/// A ~1000-vertex polygon with a 200-vertex hole.
fn polygon_with_hole(rng: &mut Rng, outer: usize) -> Polygon {
    let (hl, hn) = ring(rng, 10.0, 20.0, 2.0, 200, false);
    let hole = Polygon::new(&hl, &hn, vec![]).unwrap();
    let (la, lo) = ring(rng, 10.0, 20.0, 5.0, outer, true);
    Polygon::new(&la, &lo, vec![hole]).unwrap()
}

fn bench_polygon2d(c: &mut Criterion) {
    let mut rng = Rng(42);
    for n in [1_000usize, 10_000] {
        let poly = polygon_with_hole(&mut rng, n);
        let tree = LatLonGeometry::Polygon(poly.clone())
            .to_component2d()
            .unwrap();
        let pts: Vec<(f64, f64)> = (0..100_000)
            .map(|_| (rng.f64_in(15.0, 25.0), rng.f64_in(5.0, 15.0)))
            .collect();
        c.bench_function(&format!("polygon2d/contains_{n}v_100k_points"), |b| {
            b.iter(|| {
                let mut hits = 0u32;
                for &(x, y) in &pts {
                    hits += u32::from(tree.contains(black_box(x), black_box(y)));
                }
                black_box(hits)
            })
        });
        let boxes: Vec<(f64, f64)> = pts.iter().take(10_000).copied().collect();
        c.bench_function(&format!("polygon2d/relate_{n}v_10k_boxes"), |b| {
            b.iter(|| {
                let mut acc = 0u32;
                for &(x, y) in &boxes {
                    acc += tree.relate(x, x + 0.01, y, y + 0.01) as u32;
                }
                black_box(acc)
            })
        });
        c.bench_function(&format!("polygon2d/create_{n}v"), |b| {
            b.iter(|| {
                black_box(
                    LatLonGeometry::Polygon(poly.clone())
                        .to_component2d()
                        .unwrap(),
                )
            })
        });
    }
}

fn bench_predicates(c: &mut Criterion) {
    let mut rng = Rng(7);
    let poly = polygon_with_hole(&mut rng, 1_000);
    let tree = LatLonGeometry::Polygon(poly).to_component2d().unwrap();
    let pred = GeoEncodingUtils::create_component_predicate(tree.as_ref()).unwrap();
    let dist = GeoEncodingUtils::create_distance_predicate(10.0, 20.0, 50_000.0).unwrap();
    let enc = |rng: &mut Rng, la: (f64, f64), lo: (f64, f64)| -> Vec<(i32, i32)> {
        (0..1_000_000)
            .map(|_| {
                (
                    GeoEncodingUtils::encode_latitude(rng.f64_in(la.0, la.1)).unwrap(),
                    GeoEncodingUtils::encode_longitude(rng.f64_in(lo.0, lo.1)).unwrap(),
                )
            })
            .collect()
    };
    let poly_pts = enc(&mut rng, (5.0, 15.0), (15.0, 25.0));
    let dist_pts = enc(&mut rng, (9.5, 10.5), (19.5, 20.5));
    c.bench_function("predicate/component2d_test_1m", |b| {
        b.iter(|| {
            let mut hits = 0u32;
            for &(la, lo) in &poly_pts {
                hits += u32::from(pred.test(la, lo));
            }
            black_box(hits)
        })
    });
    c.bench_function("predicate/distance_test_1m", |b| {
        b.iter(|| {
            let mut hits = 0u32;
            for &(la, lo) in &dist_pts {
                hits += u32::from(dist.test(la, lo));
            }
            black_box(hits)
        })
    });
    let raw: Vec<(f64, f64)> = (0..1_000_000)
        .map(|_| (rng.f64_in(-90.0, 90.0), rng.f64_in(-180.0, 180.0)))
        .collect();
    c.bench_function("encoding/encode_lat_lon_1m", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for &(la, lo) in &raw {
                acc ^= GeoEncodingUtils::encode_latitude(black_box(la)).unwrap();
                acc ^= GeoEncodingUtils::encode_longitude(black_box(lo)).unwrap();
            }
            black_box(acc)
        })
    });
    c.bench_function("encoding/decode_lat_lon_1m", |b| {
        b.iter(|| {
            let mut acc = 0f64;
            for &(la, lo) in &poly_pts {
                acc += GeoEncodingUtils::decode_latitude(black_box(la));
                acc += GeoEncodingUtils::decode_longitude(black_box(lo));
            }
            black_box(acc)
        })
    });
}

fn bench_component_tree(c: &mut Criterion) {
    let mut rng = Rng(11);
    let mut geoms = Vec::new();
    for i in 0..200 {
        let lat = -60.0 + (i / 20) as f64 * 12.0;
        let lon = -150.0 + (i % 20) as f64 * 15.0;
        let (la, lo) = ring(&mut rng, lat, lon, 4.0, 20, true);
        geoms.push(LatLonGeometry::Polygon(
            Polygon::new(&la, &lo, vec![]).unwrap(),
        ));
    }
    let tree = LatLonGeometry::create(&geoms).unwrap();
    let pts: Vec<(f64, f64)> = (0..100_000)
        .map(|_| (rng.f64_in(-180.0, 180.0), rng.f64_in(-70.0, 70.0)))
        .collect();
    c.bench_function("component_tree/contains_200polys_100k", |b| {
        b.iter(|| {
            let mut hits = 0u32;
            for &(x, y) in &pts {
                hits += u32::from(tree.contains(x, y));
            }
            black_box(hits)
        })
    });
    c.bench_function("component_tree/relate_200polys_10k", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &(x, y) in pts.iter().take(10_000) {
                acc += tree.relate(x, x + 0.5, y, y + 0.5) as u32;
            }
            black_box(acc)
        })
    });
}

fn bench_parsers(c: &mut Criterion) {
    let mut rng = Rng(3);
    let poly = polygon_with_hole(&mut rng, 10_000);
    let json = format!(
        "{{\"type\": \"Polygon\", \"coordinates\": {}}}",
        poly.to_geojson()
    );
    c.bench_function("parse/geojson_10k_vertices", |b| {
        b.iter(|| black_box(Polygon::from_geojson(black_box(&json)).unwrap()))
    });
    let mut wkt = String::from("POLYGON((");
    for i in 0..poly.num_points() {
        if i > 0 {
            wkt.push_str(", ");
        }
        wkt.push_str(&format!("{} {}", poly.poly_lon(i), poly.poly_lat(i)));
    }
    wkt.push_str("))");
    c.bench_function("parse/wkt_10k_vertices", |b| {
        b.iter(|| black_box(simple_wkt_shape_parser::parse(black_box(&wkt)).unwrap()))
    });
    let p1k = polygon_with_hole(&mut rng, 1_000);
    c.bench_function("tessellate/1k_with_hole", |b| {
        b.iter(|| black_box(tessellator::tessellate(black_box(&p1k), false).unwrap()))
    });
    let p10k = Polygon::new(poly.poly_lats(), poly.poly_lons(), vec![]).unwrap();
    c.bench_function("tessellate/10k", |b| {
        b.iter(|| black_box(tessellator::tessellate(black_box(&p10k), false).unwrap()))
    });
}

fn count_cells(tree: &dyn SpatialPrefixTree, shape: &Arc<dyn Shape>, detail: i32) -> usize {
    let mut it = tree.tree_cell_iterator(shape, detail).unwrap();
    let mut n = 0;
    while it.has_next().unwrap() {
        black_box(it.next().unwrap());
        n += 1;
    }
    n
}

fn bench_prefix_trees(c: &mut Criterion) {
    let ctx: Arc<SpatialContext> = SpatialContext::geo_context();
    let circle: Arc<dyn Shape> = ctx.circle(10.0, 20.0, 1.0).unwrap();
    let quad = QuadPrefixTree::new(ctx.clone(), 12).unwrap();
    c.bench_function("prefix_tree/quad12_circle_detail8", |b| {
        b.iter(|| black_box(count_cells(&quad, &circle, 8)))
    });
    let gh = GeohashPrefixTree::new(ctx.clone(), 6).unwrap();
    c.bench_function("prefix_tree/geohash6_circle_detail4", |b| {
        b.iter(|| black_box(count_cells(&gh, &circle, 4)))
    });
    let ctx3d = SpatialContextFactory::geo3d()
        .new_spatial_context()
        .unwrap();
    let s2 = S2PrefixTree::new(ctx3d, 10, 2).unwrap();
    let mut rng = Rng(5);
    let terms: Vec<Vec<u8>> = {
        let w = s2.world_cell();
        let mut out = Vec::new();
        let mut it = w.next_level_cells(None).unwrap();
        let mut faces = Vec::new();
        while it.has_next().unwrap() {
            faces.push(it.next().unwrap());
        }
        for _ in 0..10_000 {
            let mut cell = faces[(rng.next_u64() % faces.len() as u64) as usize].clone();
            for _ in 0..5 {
                let mut kids = Vec::new();
                let mut it = cell.next_level_cells(None).unwrap();
                while it.has_next().unwrap() {
                    kids.push(it.next().unwrap());
                }
                cell = kids[(rng.next_u64() % kids.len() as u64) as usize].clone();
            }
            out.push(cell.token_bytes_no_leaf());
        }
        out
    };
    c.bench_function("prefix_tree/s2_read_cell_10k", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for t in &terms {
                acc += s2.read_cell(t).unwrap().level();
            }
            black_box(acc)
        })
    });
    let dt = DateRangePrefixTree::new(&Calendar::new_default()).unwrap();
    let mut cells: Vec<Box<dyn Cell>> = (0..10_000)
        .map(|_| {
            let ms = (rng.next_u64() % 4_000_000_000_000) as i64;
            let full = dt.to_unit_shape_millis(ms);
            let lv = 1 + (rng.next_u64() % full.level() as u64) as i32;
            Box::new(full.shape_at_level(lv)) as Box<dyn Cell>
        })
        .collect();
    cells.sort_by(|a, b| a.compare_to_no_leaf(&**b).cmp(&0));
    c.bench_function("prefix_tree/date_compare_no_leaf_10k", |b| {
        b.iter(|| {
            let mut acc = 0i32;
            for w in cells.windows(2) {
                acc += w[0].compare_to_no_leaf(&*w[1]).signum();
                acc += i32::from(w[0].is_prefix_of(&*w[1]));
            }
            black_box(acc)
        })
    });
}

/// The wide-box and very large polygon edge-tree walks, many-polygon
/// triangle tests, the rectangle's edge test, the distance-circle box
/// relation and the S2 cell walk.
fn bench_hot_paths(c: &mut Criterion) {
    let mut rng = Rng(77);
    let (la, lo) = ring(&mut rng, 10.0, 20.0, 5.0, 100_000, true);
    let big = LatLonGeometry::Polygon(Polygon::new(&la, &lo, vec![]).unwrap())
        .to_component2d()
        .unwrap();
    let pts: Vec<(f64, f64)> = (0..100_000)
        .map(|_| (rng.f64_in(15.0, 25.0), rng.f64_in(5.0, 15.0)))
        .collect();
    c.bench_function("polygon2d/contains_100kv_100k_points", |b| {
        b.iter(|| {
            let mut hits = 0u32;
            for &(x, y) in &pts {
                hits += u32::from(big.contains(black_box(x), black_box(y)));
            }
            black_box(hits)
        })
    });
    let poly = polygon_with_hole(&mut rng, 10_000);
    let tree = LatLonGeometry::Polygon(poly).to_component2d().unwrap();
    for w in [0.1f64, 1.0] {
        c.bench_function(&format!("polygon2d/relate_10kv_1k_boxes_{w}deg"), |b| {
            b.iter(|| {
                let mut acc = 0u32;
                for &(x, y) in pts.iter().take(1_000) {
                    acc += tree.relate(x, x + w, y, y + w) as u32;
                }
                black_box(acc)
            })
        });
    }

    let mut geoms = Vec::new();
    for i in 0..200 {
        let lat = -60.0 + (i / 20) as f64 * 12.0;
        let lon = -150.0 + (i % 20) as f64 * 15.0;
        let (la, lo) = ring(&mut rng, lat, lon, 4.0, 20, true);
        geoms.push(LatLonGeometry::Polygon(
            Polygon::new(&la, &lo, vec![]).unwrap(),
        ));
    }
    let multi = LatLonGeometry::create(&geoms).unwrap();
    let tris: Vec<[f64; 6]> = (0..100_000)
        .map(|_| {
            let x = rng.f64_in(-180.0, 179.0);
            let y = rng.f64_in(-70.0, 69.0);
            [
                x,
                y,
                x + rng.f64_in(0.0, 0.5),
                y,
                x,
                y + rng.f64_in(0.0, 0.5),
            ]
        })
        .collect();
    c.bench_function("component_tree/intersects_triangle_200polys_100k", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for t in &tris {
                acc += u32::from(multi.intersects_triangle(t[0], t[1], t[2], t[3], t[4], t[5]));
            }
            black_box(acc)
        })
    });

    let rect = XYGeometry::create(&[XYGeometry::Rectangle(
        XYRectangle::new(-10.0, 10.0, -10.0, 10.0).unwrap(),
    )])
    .unwrap();
    let edge_tris: Vec<[f64; 6]> = (0..1_000_000)
        .map(|_| {
            let side = rng.next_u64() % 4;
            let t = rng.f64_in(-12.0, 12.0);
            let (x, y) = match side {
                0 => (rng.f64_in(-11.0, -9.0), t),
                1 => (rng.f64_in(9.0, 11.0), t),
                2 => (t, rng.f64_in(-11.0, -9.0)),
                _ => (t, rng.f64_in(9.0, 11.0)),
            };
            [
                x,
                y,
                x + rng.f64_in(-0.5, 0.5),
                y + rng.f64_in(-0.5, 0.5),
                x + rng.f64_in(-0.5, 0.5),
                y + rng.f64_in(-0.5, 0.5),
            ]
        })
        .collect();
    c.bench_function("rectangle2d/intersects_triangle_edges_1m", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for t in &edge_tris {
                acc += u32::from(rect.intersects_triangle(t[0], t[1], t[2], t[3], t[4], t[5]));
            }
            black_box(acc)
        })
    });

    let (lat, lon, radius) = (0.0, 0.0, 50_000.0);
    let sort_key = GeoUtils::distance_query_sort_key(radius);
    let axis_lat = Rectangle::axis_lat(lat, radius);
    let cells: Vec<(f64, f64, f64)> = (0..100_000)
        .map(|_| {
            (
                rng.f64_in(-1.5, 1.5),
                rng.f64_in(-1.5, 1.5),
                rng.f64_in(0.05, 0.8),
            )
        })
        .collect();
    c.bench_function("geo_utils/relate_circle_50km_100k_boxes", |b| {
        b.iter(|| {
            let mut acc = 0u32;
            for &(y, x, w) in &cells {
                acc += GeoUtils::relate(y, y + w, x, x + w, lat, lon, sort_key, axis_lat).unwrap()
                    as u32;
            }
            black_box(acc)
        })
    });

    let ctx3d = SpatialContextFactory::geo3d()
        .new_spatial_context()
        .unwrap();
    let s2 = S2PrefixTree::new(ctx3d.clone(), 12, 2).unwrap();
    let circle3d: Arc<dyn Shape> = ctx3d.circle(10.0, 20.0, 1.0).unwrap();
    c.bench_function("prefix_tree/s2_circle_detail10", |b| {
        b.iter(|| black_box(count_cells(&s2, &circle3d, 10)))
    });
}

criterion_group!(
    benches,
    bench_hot_paths,
    bench_polygon2d,
    bench_predicates,
    bench_component_tree,
    bench_parsers,
    bench_prefix_trees
);
criterion_main!(benches);
