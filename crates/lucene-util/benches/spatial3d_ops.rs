//! Microbenchmarks for the geo3d and Spatial4j hot paths the query side
//! reaches per document or per BKD cell:
//!
//! - `GeoComplexPolygon::is_within` (`PointInGeo3DShapeQuery`'s per-doc
//!   `visit`) over random points near a ~1000-vertex polygon;
//! - `make_xyz_solid` + `get_relationship` (the per-cell `compare`) against a
//!   circle, a path and the complex polygon;
//! - `GeoStandardPath::compute_distance` (the per-doc distance sort);
//! - Spatial4j circle/rectangle relate and geohash decoding (prefix-tree
//!   cells).
//!
//! Run with: `cargo bench -p lucene-util --bench spatial3d_ops`

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use lucene_util::spatial3d::geo_circle_factory::make_geo_circle;
use lucene_util::spatial3d::geo_polygon_factory::{make_large_geo_polygon, PolygonDescription};
use lucene_util::spatial3d::geo_standard_path::GeoStandardPath;
use lucene_util::spatial3d::xyz_solid::make_xyz_solid;
use lucene_util::spatial3d::{DistanceStyle, GeoDistance, GeoPoint, GeoShape, PlanetModel};
use lucene_util::spatial4j::{geohash, Shape, SpatialContext};

/// A small deterministic xorshift generator (no `rand` dependency).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Uniform in `[0, 1)`.
    fn f(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.f()
    }
}

fn rad(d: f64) -> f64 {
    d.to_radians()
}

/// A wobbly ~`n`-vertex ring around (20N, 30E), about 10 degrees across.
fn polygon_points(pm: &PlanetModel, n: usize) -> Vec<GeoPoint> {
    (0..n)
        .map(|i| {
            let a = i as f64 / n as f64 * std::f64::consts::TAU;
            let r = 10.0 + 2.0 * (7.0 * a).sin();
            GeoPoint::from_lat_lon(pm, rad(20.0 + r * a.sin()), rad(30.0 + r * a.cos())).unwrap()
        })
        .collect()
}

/// Random surface points around the polygon's region.
fn near_points(pm: &PlanetModel, rng: &mut Rng, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|_| {
            let p =
                GeoPoint::from_lat_lon(pm, rad(rng.range(5.0, 35.0)), rad(rng.range(15.0, 45.0)))
                    .unwrap();
            [p.x, p.y, p.z]
        })
        .collect()
}

/// BKD-like cells: around a surface point near the region, with an extent
/// log-uniform in `[1e-6, 1]`.
fn cells(pm: &PlanetModel, rng: &mut Rng, n: usize) -> Vec<[f64; 6]> {
    near_points(pm, rng, n)
        .into_iter()
        .map(|[x, y, z]| {
            let s = 10f64.powf(rng.range(-6.0, 0.0));
            let (ox, oy, oz) = (rng.range(-s, 0.0), rng.range(-s, 0.0), rng.range(-s, 0.0));
            [x + ox, x + ox + s, y + oy, y + oy + s, z + oz, z + oz + s]
        })
        .collect()
}

fn bench_complex_polygon(c: &mut Criterion) {
    let pm = PlanetModel::wgs84();
    let ring = polygon_points(&pm, 1000);
    let poly = make_large_geo_polygon(&pm, &[PolygonDescription::new(ring)]).unwrap();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let pts = near_points(&pm, &mut rng, 1000);
    c.bench_function("complex_polygon_1k/is_within_1k_points", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for &[x, y, z] in &pts {
                n += u32::from(poly.is_within_xyz(black_box(x), y, z));
            }
            black_box(n)
        })
    });
    let cs = cells(&pm, &mut rng, 1000);
    let shape: &dyn GeoShape = &*poly;
    c.bench_function("complex_polygon_1k/solid_relationship_1k_cells", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for c in &cs {
                let s = make_xyz_solid(&pm, c[0], c[1], c[2], c[3], c[4], c[5]).unwrap();
                n += s.get_relationship(black_box(shape)).unwrap() as u32;
            }
            black_box(n)
        })
    });
}

fn path(pm: &Arc<PlanetModel>, n: usize) -> GeoStandardPath {
    let pts: Vec<GeoPoint> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64;
            GeoPoint::from_lat_lon(
                pm,
                rad(10.0 + 20.0 * t + 1.5 * (t * 40.0).sin()),
                rad(20.0 + 20.0 * t),
            )
            .unwrap()
        })
        .collect();
    GeoStandardPath::new(pm, rad(1.0), &pts).unwrap()
}

fn bench_path(c: &mut Criterion) {
    let pm = PlanetModel::wgs84();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let pts = near_points(&pm, &mut rng, 1000);
    for n in [10usize, 100, 1000] {
        let p = path(&pm, n);
        c.bench_function(&format!("standard_path_{n}/distance_arc_1k_points"), |b| {
            b.iter(|| {
                let mut s = 0.0;
                for &[x, y, z] in &pts {
                    let d = p.compute_distance(DistanceStyle::Arc, black_box(x), y, z);
                    if d.is_finite() {
                        s += d;
                    }
                }
                black_box(s)
            })
        });
    }
    let p = path(&pm, 100);
    let cs = cells(&pm, &mut rng, 1000);
    c.bench_function("standard_path_100/solid_relationship_1k_cells", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for c in &cs {
                let s = make_xyz_solid(&pm, c[0], c[1], c[2], c[3], c[4], c[5]).unwrap();
                n += s.get_relationship(black_box(&p)).unwrap() as u32;
            }
            black_box(n)
        })
    });
    let circle = make_geo_circle(&pm, rad(20.0), rad(30.0), rad(5.0)).unwrap();
    let circle: &dyn GeoShape = &*circle;
    c.bench_function("circle/solid_relationship_1k_cells", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for c in &cs {
                let s = make_xyz_solid(&pm, c[0], c[1], c[2], c[3], c[4], c[5]).unwrap();
                n += s.get_relationship(black_box(circle)).unwrap() as u32;
            }
            black_box(n)
        })
    });
}

fn bench_spatial4j(c: &mut Criterion) {
    let ctx = SpatialContext::geo_context();
    let mut rng = Rng(0xdead_beef_cafe_f00d);
    let circle = ctx.circle(30.0, 20.0, 5.0).unwrap();
    let rects: Vec<_> = (0..1000)
        .map(|_| {
            let w = 10f64.powf(rng.range(-3.0, 1.0));
            let x = rng.range(10.0, 50.0);
            let y = rng.range(0.0, 40.0);
            ctx.rect(x, x + w, y, y + w).unwrap()
        })
        .collect();
    c.bench_function("spatial4j/circle_relate_rect_1k", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for r in &rects {
                n += circle.relate(black_box(&**r as &dyn Shape)).unwrap() as u32;
            }
            black_box(n)
        })
    });
    c.bench_function("spatial4j/rect_relate_circle_1k", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for r in &rects {
                n += r.relate(black_box(&*circle as &dyn Shape)).unwrap() as u32;
            }
            black_box(n)
        })
    });
    // `ShapeValuesPredicate`'s per-doc `op.evaluate(docShape, queryShape)`
    // with a bounding-box operation on rectangle doc shapes.
    c.bench_function("spatial4j/bbox_intersects_evaluate_1k", |b| {
        use lucene_util::spatial_extras::query::SpatialOperation;
        b.iter(|| {
            let mut n = 0u32;
            for r in &rects {
                for op in [
                    SpatialOperation::BBoxIntersects,
                    SpatialOperation::BBoxWithin,
                ] {
                    n += u32::from(op.evaluate(black_box(&**r), &*circle).unwrap());
                }
            }
            black_box(n)
        })
    });
    let line_pts: Vec<_> = (0..20)
        .map(|i| {
            ctx.point_xy(10.0 + 2.0 * i as f64, 10.0 + (i % 3) as f64)
                .unwrap()
        })
        .collect();
    let line = ctx.line_string(&line_pts, 0.5).unwrap();
    c.bench_function("spatial4j/buffered_line_string_relate_rect_1k", |b| {
        b.iter(|| {
            let mut n = 0u32;
            for r in &rects {
                n += line.relate(black_box(&**r as &dyn Shape)).unwrap() as u32;
            }
            black_box(n)
        })
    });
    let hashes: Vec<String> = (0..1000)
        .map(|_| geohash::encode_lat_lon(rng.range(-90.0, 90.0), rng.range(-180.0, 180.0), 12))
        .collect();
    c.bench_function("spatial4j/geohash_decode_1k", |b| {
        b.iter(|| {
            let mut s = 0.0;
            for h in &hashes {
                s += geohash::decode_boundary_values(black_box(h)).unwrap()[0];
            }
            black_box(s)
        })
    });
}

criterion_group!(benches, bench_complex_polygon, bench_path, bench_spatial4j);
criterion_main!(benches);
