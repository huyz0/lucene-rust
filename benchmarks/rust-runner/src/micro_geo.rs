//! The geo benchmark pair, against `benchmarks/micro/java/GeoMicro.java`:
//! the `Tessellator`, `Component2D` construction and relations, and
//! `SloppyMath.haversinMeters`. Both sides read the differential fixtures
//! under `fixtures/data/geo/`, so they time the polygons, shapes and queries
//! the tests verify; every case prints a `#check` digest the report compares
//! across the engines before it shows a ratio.

use std::hint::black_box;
use std::time::Duration;

use lucene_util::geo::tessellator;
use lucene_util::geo::{
    Circle, Component2D, LatLonGeometry, Line, Point, Polygon, Rectangle, XYCircle, XYGeometry,
    XYLine, XYPoint, XYPolygon, XYRectangle,
};
use lucene_util::sloppy_math;

use super::measure;

const DIR: &str = "fixtures/data/geo";

/// FNV-1a over 64-bit words, identical to `GeoMicro.Fnv`.
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

fn read(name: &str) -> String {
    std::fs::read_to_string(format!("{DIR}/{name}")).unwrap_or_else(|e| panic!("{DIR}/{name}: {e}"))
}

fn unesc(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('t') => out.push('\t'),
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some(o) => out.push(o),
                None => {}
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn ring<T: std::str::FromStr>(s: &str) -> (Vec<T>, Vec<T>)
where
    T::Err: std::fmt::Debug,
{
    s.split(';')
        .map(|p| {
            let (a, b) = p.split_once(' ').unwrap();
            (a.parse::<T>().unwrap(), b.parse::<T>().unwrap())
        })
        .unzip()
}

fn polygon(body: &str) -> Polygon {
    let mut rings = body.split('|');
    let (la, lo) = ring::<f64>(rings.next().unwrap());
    let holes = rings
        .map(|r| {
            let (a, b) = ring::<f64>(r);
            Polygon::new(&a, &b, vec![]).unwrap()
        })
        .collect();
    Polygon::new(&la, &lo, holes).unwrap()
}

fn xy_polygon(body: &str) -> XYPolygon {
    let mut rings = body.split('|');
    let (x, y) = ring::<f32>(rings.next().unwrap());
    let holes = rings
        .map(|r| {
            let (a, b) = ring::<f32>(r);
            XYPolygon::new(&a, &b, vec![]).unwrap()
        })
        .collect();
    XYPolygon::new(&x, &y, holes).unwrap()
}

fn nums(s: &str) -> Vec<f64> {
    s.split(',').map(|v| v.parse().unwrap()).collect()
}

fn latlon(spec: &str) -> Vec<LatLonGeometry> {
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            match kind {
                "P:" => {
                    let v = nums(body);
                    LatLonGeometry::Point(Point::new(v[0], v[1]).unwrap())
                }
                "L:" => {
                    let (a, b) = ring::<f64>(body);
                    LatLonGeometry::Line(Line::new(&a, &b).unwrap())
                }
                "G:" => LatLonGeometry::Polygon(polygon(body)),
                "C:" => {
                    let v = nums(body);
                    LatLonGeometry::Circle(Circle::new(v[0], v[1], v[2]).unwrap())
                }
                _ => {
                    let v = nums(body);
                    LatLonGeometry::Rectangle(Rectangle::new(v[0], v[1], v[2], v[3]).unwrap())
                }
            }
        })
        .collect()
}

fn xy(spec: &str) -> Vec<XYGeometry> {
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            let v = || -> Vec<f32> { body.split(',').map(|v| v.parse().unwrap()).collect() };
            match kind {
                "P:" => {
                    let v = v();
                    XYGeometry::Point(XYPoint::new(v[0], v[1]).unwrap())
                }
                "L:" => {
                    let (a, b) = ring::<f32>(body);
                    XYGeometry::Line(XYLine::new(&a, &b).unwrap())
                }
                "G:" => XYGeometry::Polygon(xy_polygon(body)),
                "C:" => {
                    let v = v();
                    XYGeometry::Circle(XYCircle::new(v[0], v[1], v[2]).unwrap())
                }
                _ => {
                    let v = v();
                    XYGeometry::Rectangle(XYRectangle::new(v[0], v[1], v[2], v[3]).unwrap())
                }
            }
        })
        .collect()
}

pub fn bench_geo(w: Duration, m: Duration) {
    bench_tessellate(w, m);
    bench_components(w, m);
    bench_haversin(w, m);
}

fn bench_tessellate(w: Duration, m: Duration) {
    let text = read("tessellator.tsv");
    let lines: Vec<&str> = text.lines().collect();
    let mut latlon_polys = Vec::new();
    let mut xy_polys = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != "poly" || f[3] != "1" || i + 1 >= lines.len() || lines[i + 1].starts_with("ERR") {
            continue;
        }
        let spec = unesc(f[4]);
        let body = &spec[2..];
        if f[2] == "latlon" {
            latlon_polys.push(polygon(body));
        } else {
            xy_polys.push(xy_polygon(body));
        }
    }
    for check_self in [true, false] {
        let name = if check_self {
            "tessellate_latlon_checked"
        } else {
            "tessellate_latlon"
        };
        let mut d = Fnv::new();
        let mut tris = 0u64;
        for p in &latlon_polys {
            for t in tessellator::tessellate(p, check_self).unwrap() {
                tris += 1;
                for v in 0..3 {
                    d.add(i64::from(t.encoded_x(v)));
                    d.add(i64::from(t.encoded_y(v)));
                    d.add(i64::from(t.is_edge_from_polygon(v)));
                }
            }
        }
        check(name, &d, tris);
        measure(name, w, m, || {
            let mut n = 0usize;
            for p in &latlon_polys {
                n += tessellator::tessellate(black_box(p), check_self).unwrap().len();
            }
            black_box(n);
            latlon_polys.len() as u64
        });
    }
    let mut d = Fnv::new();
    let mut tris = 0u64;
    for p in &xy_polys {
        for t in tessellator::tessellate_xy(p, true).unwrap() {
            tris += 1;
            d.add(i64::from(t.encoded_x(0)));
            d.add(i64::from(t.encoded_y(2)));
        }
    }
    check("tessellate_xy_checked", &d, tris);
    measure("tessellate_xy_checked", w, m, || {
        let mut n = 0usize;
        for p in &xy_polys {
            n += tessellator::tessellate_xy(black_box(p), true).unwrap().len();
        }
        black_box(n);
        xy_polys.len() as u64
    });
}

struct Shape {
    geo: bool,
    spec: String,
    relate: Vec<[f64; 4]>,
    contains: Vec<[f64; 2]>,
    tris: Vec<[f64; 6]>,
}

fn build(s: &Shape) -> Box<dyn Component2D> {
    if s.geo {
        LatLonGeometry::create(&latlon(&s.spec)).unwrap()
    } else {
        XYGeometry::create(&xy(&s.spec)).unwrap()
    }
}

fn bench_components(w: Duration, m: Duration) {
    let text = read("component2d.tsv");
    let mut shapes: Vec<Shape> = Vec::new();
    let mut active = false;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] == "shape" {
            active = f.len() == 4;
            if active {
                shapes.push(Shape {
                    geo: f[2] == "latlon",
                    spec: unesc(f[3]),
                    relate: Vec::new(),
                    contains: Vec::new(),
                    tris: Vec::new(),
                });
            }
            continue;
        }
        if !active {
            continue;
        }
        let s = shapes.last_mut().unwrap();
        match f[1] {
            "relate" => {
                let v = nums(f[2]);
                s.relate.push([v[0], v[1], v[2], v[3]]);
            }
            "contains" => {
                let v = nums(f[2]);
                s.contains.push([v[0], v[1]]);
            }
            "itri" => {
                let v = nums(f[2]);
                s.tris.push([v[0], v[1], v[2], v[3], v[4], v[5]]);
            }
            _ => {}
        }
    }
    let comps: Vec<Box<dyn Component2D>> = shapes.iter().map(build).collect();

    measure("component_build", w, m, || {
        let mut n = 0u64;
        for s in &shapes {
            n = n.wrapping_add(build(black_box(s)).min_x().to_bits());
        }
        black_box(n);
        shapes.len() as u64
    });

    const REPS: u64 = 20;
    let mut d = Fnv::new();
    let mut q = 0u64;
    for (s, c) in shapes.iter().zip(&comps) {
        for b in &s.relate {
            d.add(c.relate(b[0], b[1], b[2], b[3]) as i64);
            q += 1;
        }
    }
    check("component_relate", &d, q);
    measure("component_relate", w, m, || {
        let mut n = 0u64;
        for _ in 0..REPS {
            for (s, c) in shapes.iter().zip(&comps) {
                for b in &s.relate {
                    n += black_box(c).relate(b[0], b[1], b[2], b[3]) as u64;
                }
            }
        }
        black_box(n);
        q * REPS
    });

    let mut d = Fnv::new();
    let mut q = 0u64;
    for (s, c) in shapes.iter().zip(&comps) {
        for p in &s.contains {
            d.add(i64::from(c.contains(p[0], p[1])));
            q += 1;
        }
    }
    check("component_contains", &d, q);
    measure("component_contains", w, m, || {
        let mut n = 0u64;
        for _ in 0..REPS {
            for (s, c) in shapes.iter().zip(&comps) {
                for p in &s.contains {
                    n += u64::from(black_box(c).contains(p[0], p[1]));
                }
            }
        }
        black_box(n);
        q * REPS
    });

    let mut d = Fnv::new();
    let mut q = 0u64;
    for (s, c) in shapes.iter().zip(&comps) {
        for t in &s.tris {
            d.add(i64::from(c.intersects_triangle(t[0], t[1], t[2], t[3], t[4], t[5])));
            q += 1;
        }
    }
    check("component_intersects_triangle", &d, q);
    measure("component_intersects_triangle", w, m, || {
        let mut n = 0u64;
        for _ in 0..REPS {
            for (s, c) in shapes.iter().zip(&comps) {
                for t in &s.tris {
                    n += u64::from(black_box(c).intersects_triangle(t[0], t[1], t[2], t[3], t[4], t[5]));
                }
            }
        }
        black_box(n);
        q * REPS
    });
}

fn bench_haversin(w: Duration, m: Duration) {
    let mut pts: Vec<[f64; 4]> = Vec::new();
    for line in read("sloppy_math.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] != "hav" {
            continue;
        }
        let v: Vec<f64> = f[1]
            .split(',')
            .map(|h| f64::from_bits(u64::from_str_radix(h, 16).unwrap()))
            .collect();
        pts.push([v[0], v[1], v[2], v[3]]);
    }
    let mut d = Fnv::new();
    for p in &pts {
        d.add(sloppy_math::haversin_meters(p[0], p[1], p[2], p[3]).to_bits() as i64);
    }
    check("haversin_meters", &d, pts.len() as u64);
    measure("haversin_meters", w, m, || {
        let mut s = 0.0;
        for p in black_box(&pts) {
            s += sloppy_math::haversin_meters(p[0], p[1], p[2], p[3]);
        }
        black_box(s);
        pts.len() as u64
    });
}
