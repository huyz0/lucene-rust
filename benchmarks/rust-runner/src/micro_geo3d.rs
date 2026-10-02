//! The geo3d benchmark pair, against `benchmarks/micro/java/Geo3dMicro.java`:
//! polygon and circle construction, `GeoArea.getRelationship` of x/y/z
//! cells against shapes (the call `PointInGeo3DShapeQuery` makes per BKD
//! node), `isWithin`, and distances. Both sides draw the same inputs from a
//! SplitMix64 stream and print a `#check` digest of integer results the
//! report compares before it shows a ratio.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use lucene_util::spatial3d::geo_area_factory::make_geo_area;
use lucene_util::spatial3d::geo_bbox_factory::make_geo_bbox;
use lucene_util::spatial3d::geo_circle_factory::make_geo_circle;
use lucene_util::spatial3d::geo_path_factory::make_geo_path;
use lucene_util::spatial3d::geo_polygon_factory::make_geo_polygon;
use lucene_util::spatial3d::{
    DistanceStyle, GeoAreaObject, GeoDistanceShape, GeoPoint, GeoShape, PlanetModel,
};
use lucene_util::strict_math;

use super::measure;

/// FNV-1a over 64-bit words, identical to `Geo3dMicro.Fnv`.
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

/// SplitMix64, identical to `Geo3dMicro.next`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    fn lat(&mut self) -> f64 {
        (self.unit() - 0.5) * std::f64::consts::PI * 0.9
    }
    fn lon(&mut self) -> f64 {
        (self.unit() * 2.0 - 1.0) * std::f64::consts::PI
    }
}

fn star(
    rng: &mut Rng,
    pm: &PlanetModel,
    clat: f64,
    clon: f64,
    n: usize,
    radius: f64,
) -> Vec<GeoPoint> {
    use std::f64::consts::PI;
    (0..n)
        .map(|i| {
            let a = PI * 2.0 * i as f64 / n as f64;
            let r = radius * (0.5 + rng.unit() * 0.5);
            let la = (clat + r * strict_math::sin(a)).clamp(-1.5, 1.5);
            let mut lo = clon + r * strict_math::cos(a);
            if lo > PI {
                lo -= 2.0 * PI;
            }
            if lo < -PI {
                lo += 2.0 * PI;
            }
            GeoPoint::from_lat_lon(pm, la, lo).unwrap()
        })
        .collect()
}

fn round(d: f64) -> i64 {
    if d.is_infinite() {
        i64::MAX
    } else {
        // Java's Math.round: floor(d + 0.5).
        (d * 1e6 + 0.5).floor() as i64
    }
}

pub fn bench_geo3d(w: Duration, m: Duration) {
    let pm = PlanetModel::wgs84();
    let mut rng = Rng(0x3D_B3_4C_11);
    let mut ring_points = Vec::new();
    for _ in 0..200 {
        let (clat, clon, radius) = (rng.lat(), rng.lon(), 0.02 + rng.unit() * 0.3);
        let n = 6 + (rng.unit() * 14.0) as usize;
        ring_points.push(star(&mut rng, &pm, clat, clon, n, radius));
    }
    let circles: Vec<[f64; 3]> = (0..2000)
        .map(|_| [rng.lat(), rng.lon(), 0.001 + rng.unit() * 0.5])
        .collect();
    let mut paths = Vec::new();
    for _ in 0..100 {
        let (mut la, mut lo) = (rng.lat(), rng.lon());
        let n = 2 + (rng.unit() * 6.0) as usize;
        let mut p = Vec::with_capacity(n);
        for _ in 0..n {
            la = (la + (rng.unit() - 0.5) * 0.2).clamp(-1.5, 1.5);
            lo = (lo + (rng.unit() - 0.5) * 0.2).clamp(-3.1, 3.1);
            p.push(GeoPoint::from_lat_lon(&pm, la, lo).unwrap());
        }
        paths.push(p);
    }
    let points: Vec<GeoPoint> = (0..4096)
        .map(|_| GeoPoint::from_lat_lon(&pm, rng.lat(), rng.lon()).unwrap())
        .collect();

    let mut shapes: Vec<Arc<dyn GeoShape>> = Vec::new();
    let mut distance_shapes: Vec<Arc<dyn GeoDistanceShape>> = Vec::new();
    for i in 0..40 {
        shapes.push(make_geo_polygon(&pm, &ring_points[i]).unwrap().unwrap());
        let c = make_geo_circle(&pm, circles[i][0], circles[i][1], circles[i][2]).unwrap();
        shapes.push(c.clone());
        distance_shapes.push(c);
        let p = make_geo_path(&pm, 0.01, &paths[i]).unwrap();
        shapes.push(p.clone());
        distance_shapes.push(p);
        let (t, l) = (circles[i][0], circles[i][1]);
        shapes.push(make_geo_bbox(&pm, (t + 0.2).min(1.5), t - 0.1, l - 0.2, l).unwrap());
    }
    let mut cells: Vec<Arc<dyn GeoAreaObject>> = Vec::new();
    for c in points.iter().take(400) {
        let scale = [1e-3, 1e-2, 1e-1, 1.0][(rng.unit() * 4.0) as usize];
        let s = scale * (0.5 + rng.unit() * 0.5);
        let (a, b, cc) = (rng.unit(), rng.unit(), rng.unit());
        cells.push(
            make_geo_area(
                &pm,
                c.x - s,
                c.x + s * a,
                c.y - s,
                c.y + s * b,
                c.z - s,
                c.z + s * cc,
            )
            .unwrap(),
        );
    }

    let mut d = Fnv::new();
    for (i, r) in ring_points.iter().enumerate() {
        let p = make_geo_polygon(&pm, r).unwrap().unwrap();
        for k in 0..8 {
            d.add(i64::from(p.is_within(&points[(i * 8 + k) % points.len()])));
        }
    }
    check("geo3d_polygon_build", &d, 200);
    measure("geo3d_polygon_build", w, m, || {
        let mut n = 0usize;
        for r in black_box(&ring_points) {
            n += make_geo_polygon(&pm, r)
                .unwrap()
                .unwrap()
                .edge_points()
                .len();
        }
        black_box(n);
        ring_points.len() as u64
    });

    let mut d = Fnv::new();
    for c in &circles {
        let g = make_geo_circle(&pm, c[0], c[1], c[2]).unwrap();
        d.add(i64::from(g.is_within(
            &points[(c[0] * 1000.0).abs() as usize % points.len()],
        )));
    }
    check("geo3d_circle_build", &d, circles.len() as u64);
    measure("geo3d_circle_build", w, m, || {
        let mut n = 0usize;
        for c in black_box(&circles) {
            n += make_geo_circle(&pm, c[0], c[1], c[2])
                .unwrap()
                .edge_points()
                .len();
        }
        black_box(n);
        circles.len() as u64
    });

    let mut d = Fnv::new();
    let mut q = 0u64;
    for s in &shapes {
        for a in &cells {
            d.add(a.get_relationship(&**s).unwrap() as i64);
            q += 1;
        }
    }
    check("geo3d_relate", &d, q);
    measure("geo3d_relate", w, m, || {
        let mut n = 0i64;
        for s in black_box(&shapes) {
            for a in &cells {
                n += a.get_relationship(&**s).unwrap() as i64;
            }
        }
        black_box(n);
        q
    });

    let mut d = Fnv::new();
    let mut q = 0u64;
    for s in &shapes {
        for p in &points {
            d.add(i64::from(s.is_within(p)));
            q += 1;
        }
    }
    check("geo3d_within", &d, q);
    measure("geo3d_within", w, m, || {
        let mut n = 0u64;
        for s in black_box(&shapes) {
            for p in &points {
                n += u64::from(s.is_within_xyz(p.x, p.y, p.z));
            }
        }
        black_box(n);
        q
    });

    let mut d = Fnv::new();
    let mut q = 0u64;
    for s in &distance_shapes {
        for p in &points {
            d.add(round(s.compute_distance(DistanceStyle::Arc, p.x, p.y, p.z)));
            d.add(round(s.compute_outside_distance(
                DistanceStyle::Arc,
                p.x,
                p.y,
                p.z,
            )));
            q += 1;
        }
    }
    check("geo3d_distance", &d, q);
    measure("geo3d_distance", w, m, || {
        let mut t = 0.0;
        for s in black_box(&distance_shapes) {
            for p in &points {
                t += s.compute_outside_distance(DistanceStyle::Arc, p.x, p.y, p.z);
                let v = s.compute_distance(DistanceStyle::Arc, p.x, p.y, p.z);
                if v != f64::INFINITY {
                    t += v;
                }
            }
        }
        black_box(t);
        q
    });
}
