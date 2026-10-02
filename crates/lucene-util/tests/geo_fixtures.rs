//! Differential tests for `lucene_util::geo`, `sloppy_math` and
//! `strict_math` against real Lucene 10.5.0: replays the files
//! `fixtures/src/GenGeo.java` writes under `fixtures/data/geo/` and demands
//! the same bits, the same relation for every query, and the same error
//! message wherever Java throws.

use std::path::PathBuf;

use lucene_util::geo::{
    Circle, Component2D, GeoEncodingUtils, GeoError, GeoUtils, LatLonGeometry, Line, Point,
    Polygon, Rectangle, WithinRelation, XYCircle, XYEncodingUtils, XYGeometry, XYLine, XYPoint,
    XYPolygon, XYRectangle,
};
use lucene_util::{sloppy_math, strict_math};

fn fixture(name: &str) -> String {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/geo")
        .join(name);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn hd(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("hex {s:?}")))
}

fn hf(s: &str) -> f32 {
    f32::from_bits(u32::from_str_radix(s, 16).unwrap_or_else(|_| panic!("hex {s:?}")))
}

fn hds(s: &str) -> Vec<f64> {
    s.split(',').map(hd).collect()
}

fn bits(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

fn fbits(v: f32) -> String {
    format!("{:x}", v.to_bits())
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

/// `ERR <class> <message>` as Java wrote it, for a Rust error.
fn err_cols(e: &GeoError) -> String {
    format!(
        "ERR\t{}\t{}",
        e.java_class(),
        e.to_string()
            .replace('\\', "\\\\")
            .replace('\t', "\\t")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    )
}

#[test]
fn strict_math_matches_jdk() {
    let mut n = 0;
    for line in fixture("strict_math.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "sweep_sincos" => {
                let pio2_hi = f64::from_bits(0x3FF9_21FB_5440_0000);
                let pio2_lo = f64::from_bits(0x3DD0_B461_1A62_6331);
                let (hi, lo) = ((4.0 * pio2_hi) / 2048.0, (4.0 * pio2_lo) / 2048.0);
                let mut h = Fnv::new();
                for i in 0..2049 {
                    let a = f64::from(i) * hi + f64::from(i) * lo;
                    h.add(strict_math::sin(a));
                    h.add(strict_math::cos(a));
                }
                assert_eq!(format!("{:x}", h.0), f[1], "SloppyMath sin/cos table sweep");
            }
            "sweep_asin" => {
                let max = strict_math::sin(73.0f64.to_radians());
                let delta = max / 8192.0;
                let mut h = Fnv::new();
                for i in 0..8193 {
                    h.add(strict_math::asin(f64::from(i) * delta));
                }
                assert_eq!(format!("{:x}", h.0), f[1], "SloppyMath asin table sweep");
            }
            op => {
                let x = hd(f[1]);
                let got = match op {
                    "sin" => strict_math::sin(x),
                    "cos" => strict_math::cos(x),
                    "asin" => strict_math::asin(x),
                    "acos" => strict_math::acos(x),
                    _ => panic!("{op}"),
                };
                // fdlibm's out-of-domain result is `(x-x)/(x-x)`, whose NaN
                // sign is the CPU's default (negative on x86-64, positive on
                // aarch64) -- and Java leaves NaN bits unspecified -- so a NaN
                // only has to be a NaN; every other value is bit for bit.
                let want = u64::from_str_radix(f[2], 16).unwrap();
                if f64::from_bits(want).is_nan() {
                    assert!(got.is_nan(), "StrictMath.{op}({x:e}) = {got:e}, want NaN");
                } else {
                    assert_eq!(bits(got), f[2], "StrictMath.{op}({x:e})");
                }
            }
        }
        n += 1;
    }
    assert!(n > 6000, "{n}");
}

/// Distance in units of the last place between two finite doubles.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |v: f64| {
        let b = v.to_bits() as i64;
        if b < 0 {
            i64::MIN - b
        } else {
            b
        }
    };
    key(a).abs_diff(key(b))
}

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn add(&mut self, v: f64) {
        self.0 ^= v.to_bits();
        self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
    }
}

#[test]
fn sloppy_math_matches_lucene() {
    let mut n = 0;
    for line in fixture("sloppy_math.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "hav" => {
                let a = hds(f[1]);
                assert_eq!(
                    bits(sloppy_math::haversin_sort_key(a[0], a[1], a[2], a[3])),
                    f[2],
                    "haversinSortKey{a:?}"
                );
                assert_eq!(
                    bits(sloppy_math::haversin_meters(a[0], a[1], a[2], a[3])),
                    f[3],
                    "haversinMeters{a:?}"
                );
            }
            "cos" | "sin" => {
                let x = hd(f[1]);
                let got = if f[0] == "cos" {
                    sloppy_math::cos(x)
                } else {
                    sloppy_math::sin(x)
                };
                if x.abs() > 4e6 {
                    // past the table range SloppyMath calls Math.cos, a
                    // HotSpot intrinsic: within an ulp (see sloppy_math.rs)
                    assert!(ulps(got, hd(f[2])) <= 1, "{} {x:e}", f[0]);
                } else {
                    assert_eq!(bits(got), f[2], "{} {x:e}", f[0]);
                }
            }
            "asin" => assert_eq!(bits(sloppy_math::asin(hd(f[1]))), f[2], "asin {}", f[1]),
            "hmkey" => assert_eq!(
                bits(sloppy_math::haversin_meters_from_sort_key(hd(f[1]))),
                f[2],
                "haversinMeters({})",
                f[1]
            ),
            other => panic!("{other}"),
        }
        n += 1;
    }
    assert!(n > 6000, "{n}");
}

#[test]
fn encoding_matches_lucene() {
    for line in fixture("encoding.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "lat" | "lon" => {
                let v = hd(f[1]);
                let (floor, ceil) = if f[0] == "lat" {
                    (
                        GeoEncodingUtils::encode_latitude(v),
                        GeoEncodingUtils::encode_latitude_ceil(v),
                    )
                } else {
                    (
                        GeoEncodingUtils::encode_longitude(v),
                        GeoEncodingUtils::encode_longitude_ceil(v),
                    )
                };
                let got = match (floor, ceil) {
                    (Ok(a), Ok(b)) => format!("{a}\t{b}"),
                    (Err(e), _) | (_, Err(e)) => err_cols(&e),
                };
                assert_eq!(got, f[2..].join("\t"), "{} {v:e}", f[0]);
            }
            "dec" => {
                let e: i32 = f[1].parse().unwrap();
                assert_eq!(bits(GeoEncodingUtils::decode_latitude(e)), f[2]);
                assert_eq!(bits(GeoEncodingUtils::decode_longitude(e)), f[3]);
            }
            "xy" => {
                let v = hf(f[1]);
                let got = match XYEncodingUtils::encode(v) {
                    Ok(e) => format!("{e}\t{}", fbits(XYEncodingUtils::decode(e))),
                    Err(e) => err_cols(&e),
                };
                assert_eq!(got, f[2..].join("\t"), "xy {v:e}");
            }
            other => panic!("{other}"),
        }
    }
}

#[test]
fn geo_utils_match_lucene() {
    let mut pred: Option<lucene_util::geo::DistancePredicate> = None;
    let (mut axis_exact, mut axis_total) = (0usize, 0usize);
    for line in fixture("geo_utils.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "dqsk" => assert_eq!(
                bits(GeoUtils::distance_query_sort_key(hd(f[1]))),
                f[2],
                "distanceQuerySortKey({})",
                f[1]
            ),
            "fpd" => {
                let a = hds(f[1]);
                let got = match Rectangle::from_point_distance(a[0], a[1], a[2]) {
                    Ok(r) => format!(
                        "{},{},{},{}",
                        bits(r.min_lat),
                        bits(r.max_lat),
                        bits(r.min_lon),
                        bits(r.max_lon)
                    ),
                    Err(e) => err_cols(&e),
                };
                assert_eq!(got, f[2..f.len() - 1].join("\t"), "fromPointDistance{a:?}");
                // axisLat goes through HotSpot's Math.cos intrinsic, which
                // no portable code reproduces bit for bit (see rectangle.rs).
                let want = hd(f[f.len() - 1]);
                let got = Rectangle::axis_lat(a[0], a[2]);
                assert!(ulps(got, want) <= 2, "axisLat{a:?}: {got:e} vs {want:e}");
                axis_exact += usize::from(got == want);
                axis_total += 1;
            }
            "relate" => {
                let a = hds(f[1]);
                let got = match GeoUtils::relate(a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]) {
                    Ok(r) => (r as u8).to_string(),
                    Err(e) => err_cols(&e),
                };
                assert_eq!(got, f[2..].join("\t"), "relate{a:?}");
            }
            "seg" => {
                let p = hds(f[1]);
                let got = format!(
                    "{}\t{}\t{}\t{}",
                    GeoUtils::orient(p[0], p[1], p[2], p[3], p[4], p[5]),
                    u8::from(GeoUtils::line_crosses_line(
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                    )),
                    u8::from(GeoUtils::line_overlap_line(
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                    )),
                    u8::from(GeoUtils::line_crosses_line_with_boundary(
                        p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
                    )),
                );
                assert_eq!(got, f[2..].join("\t"), "segments {p:?}");
            }
            "xyfpd" => {
                let a: Vec<f32> = f[1].split(',').map(hf).collect();
                let got = match XYRectangle::from_point_distance(a[0], a[1], a[2]) {
                    Ok(r) => format!(
                        "{},{},{},{}",
                        fbits(r.min_x),
                        fbits(r.max_x),
                        fbits(r.min_y),
                        fbits(r.max_y)
                    ),
                    Err(e) => err_cols(&e),
                };
                assert_eq!(got, f[2..].join("\t"), "XYRectangle.fromPointDistance{a:?}");
            }
            "dpred" => {
                let a = hds(f[1]);
                pred = Some(GeoEncodingUtils::create_distance_predicate(a[0], a[1], a[2]).unwrap());
            }
            "t" => {
                let (lat, lon) = f[1].split_once(',').unwrap();
                let (lat, lon): (i32, i32) = (lat.parse().unwrap(), lon.parse().unwrap());
                let got = pred.as_ref().unwrap().test(lat, lon);
                assert_eq!(
                    u8::from(got).to_string(),
                    f[2],
                    "distance predicate ({lat},{lon})"
                );
            }
            other => panic!("{other}"),
        }
    }
    // The intrinsic and StrictMath agree almost always.
    assert!(
        axis_exact * 100 >= axis_total * 99,
        "{axis_exact}/{axis_total}"
    );
}

// ---------------------------------------------------------------- shapes

fn nums(s: &str) -> Vec<f64> {
    s.split(',').map(|v| v.parse::<f64>().unwrap()).collect()
}

fn ring_d(s: &str) -> (Vec<f64>, Vec<f64>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for p in s.split(';') {
        let (x, y) = p.split_once(' ').unwrap();
        a.push(x.parse().unwrap());
        b.push(y.parse().unwrap());
    }
    (a, b)
}

fn ring_f(s: &str) -> (Vec<f32>, Vec<f32>) {
    let mut a = Vec::new();
    let mut b = Vec::new();
    for p in s.split(';') {
        let (x, y) = p.split_once(' ').unwrap();
        a.push(x.parse().unwrap());
        b.push(y.parse().unwrap());
    }
    (a, b)
}

/// A spec from `GeoCorpus.spec` back into lat/lon geometries.
pub fn parse_latlon(spec: &str) -> Result<Vec<LatLonGeometry>, GeoError> {
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            Ok(match kind {
                "P:" => {
                    let v = nums(body);
                    LatLonGeometry::Point(Point::new(v[0], v[1])?)
                }
                "L:" => {
                    let (lats, lons) = ring_d(body);
                    LatLonGeometry::Line(Line::new(&lats, &lons)?)
                }
                "G:" => LatLonGeometry::Polygon(parse_polygon(body)?),
                "C:" => {
                    let v = nums(body);
                    LatLonGeometry::Circle(Circle::new(v[0], v[1], v[2])?)
                }
                "R:" => {
                    let v = nums(body);
                    LatLonGeometry::Rectangle(Rectangle::new(v[0], v[1], v[2], v[3])?)
                }
                other => panic!("{other}"),
            })
        })
        .collect()
}

/// A `G:` body into a [`Polygon`].
pub fn parse_polygon(body: &str) -> Result<Polygon, GeoError> {
    let mut rings = body.split('|');
    let (lats, lons) = ring_d(rings.next().unwrap());
    let holes = rings
        .map(|r| {
            let (la, lo) = ring_d(r);
            Polygon::new(&la, &lo, vec![])
        })
        .collect::<Result<Vec<_>, _>>()?;
    Polygon::new(&lats, &lons, holes)
}

/// A `G:` body into an [`XYPolygon`].
pub fn parse_xy_polygon(body: &str) -> Result<XYPolygon, GeoError> {
    let mut rings = body.split('|');
    let (x, y) = ring_f(rings.next().unwrap());
    let holes = rings
        .map(|r| {
            let (a, b) = ring_f(r);
            XYPolygon::new(&a, &b, vec![])
        })
        .collect::<Result<Vec<_>, _>>()?;
    XYPolygon::new(&x, &y, holes)
}

fn parse_xy(spec: &str) -> Result<Vec<XYGeometry>, GeoError> {
    spec.split(" + ")
        .map(|g| {
            let (kind, body) = g.split_at(2);
            let fl = |s: &str| -> Vec<f32> { s.split(',').map(|v| v.parse().unwrap()).collect() };
            Ok(match kind {
                "P:" => {
                    let v = fl(body);
                    XYGeometry::Point(XYPoint::new(v[0], v[1])?)
                }
                "L:" => {
                    let (x, y) = ring_f(body);
                    XYGeometry::Line(XYLine::new(&x, &y)?)
                }
                "G:" => XYGeometry::Polygon(parse_xy_polygon(body)?),
                "C:" => {
                    let v = fl(body);
                    XYGeometry::Circle(XYCircle::new(v[0], v[1], v[2])?)
                }
                "R:" => {
                    let v = fl(body);
                    XYGeometry::Rectangle(XYRectangle::new(v[0], v[1], v[2], v[3])?)
                }
                other => panic!("{other}"),
            })
        })
        .collect()
}

fn within(w: Result<WithinRelation, GeoError>) -> String {
    match w {
        Ok(WithinRelation::Candidate) => "C".into(),
        Ok(WithinRelation::NotWithin) => "N".into(),
        Ok(WithinRelation::Disjoint) => "D".into(),
        Err(e) => err_cols(&e),
    }
}

fn d(v: f64) -> String {
    lucene_util::geo::java_double_string(v)
}

#[test]
fn component2d_matches_lucene() {
    let text = fixture("component2d.tsv");
    let mut current: Option<(String, Box<dyn Component2D>)> = None;
    let mut queries = 0;
    let mut shapes = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if f[0] == "shape" {
            let spec = unesc(f[3]);
            let built = if f[2] == "latlon" {
                parse_latlon(&spec).and_then(|g| LatLonGeometry::create(&g))
            } else {
                parse_xy(&spec).and_then(|g| XYGeometry::create(&g))
            };
            match built {
                Ok(c) => {
                    assert_eq!(f.len(), 4, "shape {} built in Rust, failed in Java", f[1]);
                    current = Some((format!("shape {}", f[1]), c));
                }
                Err(e) => {
                    assert_eq!(err_cols(&e), f[4..].join("\t"), "shape {}", f[1]);
                    current = None;
                }
            }
            shapes += 1;
            continue;
        }
        let (name, c) = current.as_ref().expect("query without a shape");
        let c: &dyn Component2D = c.as_ref();
        let op = f[1];
        let a = if f[2].is_empty() {
            Vec::new()
        } else {
            nums(f[2])
        };
        let want = f[3..].join("\t");
        let b = |x: bool| u8::from(x).to_string();
        let got = match op {
            "bounds" => format!(
                "{},{},{},{}",
                d(c.min_x()),
                d(c.max_x()),
                d(c.min_y()),
                d(c.max_y())
            ),
            "contains" => b(c.contains(a[0], a[1])),
            "relate" => (c.relate(a[0], a[1], a[2], a[3]) as u8).to_string(),
            "iline" => b(c.intersects_line(a[0], a[1], a[2], a[3])),
            "cline" => b(c.contains_line(a[0], a[1], a[2], a[3])),
            "wline" => within(c.within_line(a[0], a[1], a[4] == 1.0, a[2], a[3])),
            "itri" => b(c.intersects_triangle(a[0], a[1], a[2], a[3], a[4], a[5])),
            "ctri" => b(c.contains_triangle(a[0], a[1], a[2], a[3], a[4], a[5])),
            "wtri" => within(c.within_triangle(
                a[0],
                a[1],
                a[6] == 1.0,
                a[2],
                a[3],
                a[7] == 1.0,
                a[4],
                a[5],
                a[8] == 1.0,
            )),
            "wpoint" => within(c.within_point(a[0], a[1])),
            "pred" => {
                let p = GeoEncodingUtils::create_component_predicate(c).unwrap();
                b(p.test(a[0] as i32, a[1] as i32))
            }
            "pred_err" => match GeoEncodingUtils::create_component_predicate(c) {
                Ok(_) => "built".into(),
                Err(e) => err_cols(&e),
            },
            other => panic!("{other}"),
        };
        assert_eq!(got, want, "{name}: {op}({})", f[2]);
        queries += 1;
    }
    assert!(
        shapes > 90 && queries > 9000,
        "{shapes} shapes, {queries} queries"
    );
}

// ---------------------------------------------------------------- tessellator

#[test]
fn tessellator_matches_lucene() {
    use lucene_util::geo::tessellator::{self, Triangle};
    let text = fixture("tessellator.tsv");
    let mut lines = text.lines().peekable();
    let (mut polys, mut tris_total, mut errs) = (0, 0, 0);
    while let Some(line) = lines.next() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f[0], "poly", "{line}");
        let check = f[3] == "1";
        let spec = unesc(f[4]);
        let body = spec.strip_prefix("G:").expect("polygon spec");
        let got: Result<Vec<Triangle>, GeoError> = if f[2] == "latlon" {
            tessellator::tessellate(&parse_polygon(body).unwrap(), check)
        } else {
            tessellator::tessellate_xy(&parse_xy_polygon(body).unwrap(), check)
        };
        let mut want = Vec::new();
        while let Some(next) = lines.peek() {
            if next.starts_with("poly\t") {
                break;
            }
            want.push(lines.next().unwrap().to_string());
        }
        let got: Vec<String> = match got {
            Ok(tris) => tris
                .iter()
                .map(|t| {
                    let mut s = String::from("tri\t");
                    for v in 0..3 {
                        s.push_str(&format!("{},{},", t.encoded_x(v), t.encoded_y(v)));
                    }
                    s.push_str(
                        &(0..3)
                            .map(|v| u8::from(t.is_edge_from_polygon(v)).to_string())
                            .collect::<Vec<_>>()
                            .join(","),
                    );
                    s
                })
                .collect(),
            Err(e) => vec![err_cols(&e)],
        };
        if want.len() == 1 && want[0].starts_with("ERR") {
            errs += 1;
        }
        tris_total += want.len();
        assert_eq!(got, want, "polygon {} ({} check={check})", f[1], f[2]);
        polys += 1;
    }
    assert!(
        polys > 300 && tris_total > 8_000 && errs > 50,
        "{polys} {tris_total} {errs}"
    );
}

// ---------------------------------------------------------------- parsers

fn dump_ring(lats: &[f64], lons: &[f64]) -> String {
    lats.iter()
        .zip(lons)
        .map(|(a, b)| format!("{} {}", bits(*a), bits(*b)))
        .collect::<Vec<_>>()
        .join(";")
}

fn dump_polygon(p: &Polygon) -> String {
    let mut s = format!("G({}", dump_ring(p.poly_lats(), p.poly_lons()));
    for h in p.holes() {
        s.push('|');
        s.push_str(&dump_ring(h.poly_lats(), h.poly_lons()));
    }
    s.push(')');
    s
}

fn dump_list<T>(tag: &str, items: &[Option<T>], f: impl Fn(&T) -> String) -> String {
    let inner: Vec<String> = items
        .iter()
        .map(|i| i.as_ref().map_or_else(|| "null".to_string(), &f))
        .collect();
    format!("{tag}[{}]", inner.join(","))
}

fn dump_wkt(g: &Option<lucene_util::geo::simple_wkt_shape_parser::WktGeometry>) -> String {
    use lucene_util::geo::simple_wkt_shape_parser::WktGeometry as W;
    let point = |p: &[f64; 2]| format!("P({},{})", bits(p[0]), bits(p[1]));
    match g {
        None => "null".into(),
        Some(W::Point(p)) => point(p),
        Some(W::MultiPoint(v)) => {
            format!("MP[{}]", v.iter().map(point).collect::<Vec<_>>().join(","))
        }
        Some(W::Line(l)) => format!("L({})", dump_ring(l.lats(), l.lons())),
        Some(W::MultiLine(v)) => dump_list("ML", v, |l: &Line| {
            format!("L({})", dump_ring(l.lats(), l.lons()))
        }),
        Some(W::Polygon(p)) => dump_polygon(p),
        Some(W::MultiPolygon(v)) => dump_list("MG", v, dump_polygon),
        Some(W::Envelope(r)) => format!(
            "R({},{},{},{})",
            bits(r.min_lat),
            bits(r.max_lat),
            bits(r.min_lon),
            bits(r.max_lon)
        ),
        Some(W::GeometryCollection(v)) => {
            let inner: Vec<String> = v.iter().map(|g| dump_wkt(&g.clone())).collect();
            format!("GC[{}]", inner.join(","))
        }
    }
}

fn err_with_offset(e: &GeoError) -> String {
    match e {
        GeoError::Parse { offset, .. } => format!("{}\t{offset}", err_cols(e)),
        _ => err_cols(e),
    }
}

#[test]
fn wkt_parser_matches_lucene() {
    use lucene_util::geo::simple_wkt_shape_parser::{self, ShapeType};
    let mut n = 0;
    for line in fixture("wkt.tsv").lines() {
        let (input, want) = line.split_once('\t').unwrap();
        let input = unesc(input);
        let got = if let Some(typed) = input.strip_prefix('@') {
            let (ty, wkt) = typed.split_once(' ').unwrap();
            let ty = ShapeType::for_name(&ty.to_lowercase()).unwrap();
            simple_wkt_shape_parser::parse_expected_type(wkt, Some(ty))
        } else {
            simple_wkt_shape_parser::parse(&input)
        };
        let got = match got {
            Ok(g) => dump_wkt(&g),
            Err(e) => err_with_offset(&e),
        };
        assert_eq!(got, want, "WKT {input:?}");
        n += 1;
    }
    assert!(n > 700, "{n}");
}

#[test]
fn geojson_parser_matches_lucene() {
    let mut n = 0;
    for line in fixture("geojson.tsv").lines() {
        let (input, want) = line.split_once('\t').unwrap();
        let input = unesc(input);
        let got = match Polygon::from_geojson(&input) {
            Ok(v) => format!(
                "MG[{}]",
                v.iter().map(dump_polygon).collect::<Vec<_>>().join(",")
            ),
            Err(e) => err_with_offset(&e),
        };
        assert_eq!(got, want, "GeoJSON {input:?}");
        n += 1;
    }
    assert!(n > 450, "{n}");
}
