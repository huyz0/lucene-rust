//! `spatial3d` differentially against Lucene 10.5.0:
//! `fixtures/src/GenGeo3d.java` built a seeded corpus of shapes of every kind
//! on four planet models and recorded what Lucene answers for each (see its
//! class doc for the record types). This test rebuilds every shape from the
//! same inputs through the same factory and compares every answer -- bit for
//! bit for doubles (a NaN as a NaN, whatever its sign or payload), exactly
//! for booleans, relationships, class names, exceptions and serialized bytes.

use std::collections::HashMap;
use std::sync::Arc;

use lucene_util::spatial3d::errors::catch;
use lucene_util::spatial3d::geo_area_factory::{make_geo_area, make_geo_area_lat_lon};
use lucene_util::spatial3d::geo_bbox_factory::make_geo_bbox;
use lucene_util::spatial3d::geo_circle_factory::{make_exact_geo_circle, make_geo_circle};
use lucene_util::spatial3d::geo_degenerate_point::GeoDegeneratePoint;
use lucene_util::spatial3d::geo_path_factory::make_geo_path;
use lucene_util::spatial3d::geo_polygon_factory::{
    make_geo_concave_polygon, make_geo_convex_polygon, make_geo_polygon_from_description,
    make_large_geo_polygon, PolygonDescription,
};
use lucene_util::spatial3d::geo_s2_shape::make_geo_s2_shape;
use lucene_util::spatial3d::serializable::Input;
use lucene_util::spatial3d::standard_objects::{
    class_name, read_planet_object, write_planet_object, StandardObject,
};
use lucene_util::spatial3d::xyz_solid::{self, make_xyz_solid};
use lucene_util::spatial3d::*;

/// The committed corpus, or `GEO3D_FIXTURES` -- a directory a generator run
/// with another `-Dgeo3d.seed` wrote, for a wider one-off sweep.
fn root() -> std::path::PathBuf {
    match std::env::var_os("GEO3D_FIXTURES") {
        Some(dir) => dir.into(),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo3d"),
    }
}

fn d(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap_or_else(|_| panic!("bad double {s}")))
}

fn h(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

/// Equal as Java's raw bits, except any NaN equals any NaN.
fn same(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

fn opt(v: Option<f64>) -> String {
    v.map_or("null".into(), h)
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// A double, or the exception computing it raised, as the generator writes
/// it.
fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

fn num(v: Result<f64>) -> String {
    match v {
        Ok(v) => h(v),
        Err(e) => err_text(&e),
    }
}

/// Compares a recorded double (or exception) with a computed one.
fn same_num(want: &[&str], got: &Result<f64>) -> (bool, usize) {
    if want[0] == "ERR" {
        let w = want[..3].join("\t");
        (matches!(got, Err(e) if err_text(e) == w), 3)
    } else {
        (matches!(got, Ok(v) if same(d(want[0]), *v)), 1)
    }
}

/// Compares a recorded double field with a computed one.
fn same_str(want: &str, got: f64) -> bool {
    if want == "null" {
        return false;
    }
    same(d(want), got)
}

/// `ERR\tclass\tmessage` as the generator writes it.
fn err_text(e: &Error) -> String {
    format!(
        "ERR\t{}\t{}",
        e.java_class(),
        e.to_string().replace('\t', "\\t").replace('\n', "\\n")
    )
}

/// An exception matches by class and message -- by class alone for a
/// `NullPointerException`, whose message is the JVM's (see
/// `Error::NullPointer`).
fn same_err(got: &str, want: &str) -> bool {
    const NPE: &str = "ERR\tjava.lang.NullPointerException\t";
    got == want || (got.starts_with(NPE) && want.starts_with(NPE))
}

fn style(s: &str) -> DistanceStyle {
    match s {
        "ARC" => DistanceStyle::Arc,
        "LINEAR" => DistanceStyle::Linear,
        "LINEAR_SQUARED" => DistanceStyle::LinearSquared,
        "NORMAL" => DistanceStyle::Normal,
        "NORMAL_SQUARED" => DistanceStyle::NormalSquared,
        other => panic!("style {other}"),
    }
}

fn rel_name(r: GeoAreaRelationship) -> &'static str {
    match r {
        GeoAreaRelationship::Contains => "CONTAINS",
        GeoAreaRelationship::Within => "WITHIN",
        GeoAreaRelationship::Overlaps => "OVERLAPS",
        GeoAreaRelationship::Disjoint => "DISJOINT",
    }
}

/// A rebuilt shape, through every interface its Java class implements.
#[derive(Clone)]
struct Handle {
    planet: Arc<dyn PlanetObject>,
    membership: Arc<dyn Membership>,
    shape: Option<Arc<dyn GeoShape>>,
    area: Option<Arc<dyn GeoArea>>,
    outside: Option<Arc<dyn GeoMembershipShape>>,
    distance: Option<Arc<dyn GeoDistanceShape>>,
    path: Option<Arc<dyn GeoPath>>,
    sizeable: Option<Arc<dyn GeoSizeable>>,
    bbox: Option<Arc<dyn GeoBBox>>,
}

impl Handle {
    fn bbox(b: Arc<dyn GeoBBox>) -> Handle {
        Handle {
            planet: b.clone(),
            membership: b.clone(),
            shape: Some(b.clone()),
            area: Some(b.clone()),
            outside: Some(b.clone()),
            distance: None,
            path: None,
            sizeable: Some(b.clone()),
            bbox: Some(b),
        }
    }

    fn distance(c: Arc<dyn GeoDistanceShape>, sizeable: Option<Arc<dyn GeoSizeable>>) -> Handle {
        Handle {
            planet: c.clone(),
            membership: c.clone(),
            shape: Some(c.clone()),
            area: Some(c.clone()),
            outside: Some(c.clone()),
            distance: Some(c),
            path: None,
            sizeable,
            bbox: None,
        }
    }

    fn path(p: Arc<dyn GeoPath>) -> Handle {
        let mut h = Handle::distance(p.clone(), None);
        h.path = Some(p);
        h
    }

    fn solid(s: Arc<dyn xyz_solid::XYZSolid>) -> Handle {
        Handle {
            planet: s.clone(),
            membership: s.clone(),
            shape: None,
            area: Some(s),
            outside: None,
            distance: None,
            path: None,
            sizeable: None,
            bbox: None,
        }
    }

    fn area_shape(s: Arc<dyn GeoAreaShape>) -> Handle {
        Handle {
            planet: s.clone(),
            membership: s.clone(),
            shape: Some(s.clone()),
            area: Some(s.clone()),
            outside: Some(s),
            distance: None,
            path: None,
            sizeable: None,
            bbox: None,
        }
    }

    fn point(p: Arc<GeoDegeneratePoint>) -> Handle {
        let mut h = Handle::bbox(p.clone());
        h.distance = Some(p);
        h
    }
}

/// A factory's `GeoBBox`, with the capabilities of its runtime class (a
/// degenerate point is also a distance shape).
fn bbox_handle(b: Arc<dyn GeoBBox>) -> Handle {
    match b.as_any().downcast_ref::<GeoDegeneratePoint>() {
        Some(p) => Handle::point(Arc::new(p.clone())),
        None => Handle::bbox(b),
    }
}

/// A `GeoAreaFactory` result, with the capabilities of its runtime class:
/// serialized and read back, as the reader returns the most specific
/// interface (the bytes are what the `SH` check compares).
fn area_handle(a: Arc<dyn GeoAreaObject>) -> Handle {
    let mut bytes = Vec::new();
    write_planet_object(&mut bytes, &*a).unwrap();
    let back = read_planet_object(&mut Input::new(&bytes)).unwrap();
    let mut h = match back {
        StandardObject::Solid(s) => Handle::solid(s),
        StandardObject::BBox(b) => bbox_handle(b),
        StandardObject::PointShape(p) => {
            let p = p
                .as_any()
                .downcast_ref::<GeoDegeneratePoint>()
                .unwrap()
                .clone();
            Handle::point(Arc::new(p))
        }
        _ => panic!("GeoAreaFactory made a {}", back.class_name()),
    };
    h.planet = a;
    h
}

/// A factory's `GeoCircle`, with the capabilities of its runtime class.
fn circle_handle(c: Arc<dyn GeoCircle>) -> Handle {
    if let Some(p) = c.as_any().downcast_ref::<GeoDegeneratePoint>() {
        return Handle::point(Arc::new(p.clone()));
    }
    Handle::distance(c.clone(), Some(c))
}

/// The `GeoPoint`s of `n lat lon ...` at `tokens[*at..]`.
fn read_points(tokens: &[&str], at: &mut usize, pm: &PlanetModel) -> Result<Vec<GeoPoint>> {
    let n: usize = tokens[*at].parse().unwrap();
    *at += 1;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let (lat, lon) = (d(tokens[*at]), d(tokens[*at + 1]));
        *at += 2;
        out.push(GeoPoint::from_lat_lon(pm, lat, lon));
    }
    // Java builds every point before the factory runs; the first failure
    // is the exception.
    out.into_iter().collect()
}

/// A `PolygonDescription` written by `GenGeo3d.desc`: points, a hole
/// count, then each hole as a description.
fn read_description(
    tokens: &[&str],
    at: &mut usize,
    pm: &PlanetModel,
) -> Result<PolygonDescription> {
    let points = read_points(tokens, at, pm)?;
    let n: usize = tokens[*at].parse().unwrap();
    *at += 1;
    let mut holes = Vec::with_capacity(n);
    for _ in 0..n {
        holes.push(read_description(tokens, at, pm)?);
    }
    Ok(PolygonDescription::with_holes(points, holes))
}

/// Builds the shape a `SH` line describes; `None` for a kind not ported.
fn build(kind: &str, args: &[&str], pm: &Arc<PlanetModel>) -> Option<Result<Option<Handle>>> {
    let f: Vec<f64> = args
        .iter()
        .map(|a| {
            if a.len() == 16 || a.contains(|c: char| c.is_ascii_alphabetic()) {
                d(a)
            } else {
                0.0
            }
        })
        .collect();
    Some(match kind {
        "path" => {
            let n: usize = args[1].parse().unwrap();
            let pts: Vec<GeoPoint> = (0..n)
                .map(|i| {
                    GeoPoint::from_lat_lon(pm, d(args[2 + 2 * i]), d(args[3 + 2 * i])).unwrap()
                })
                .collect();
            make_geo_path(pm, f[0], &pts).map(|p| Some(Handle::path(p)))
        }
        "circle" => make_geo_circle(pm, f[0], f[1], f[2]).map(|c| Some(circle_handle(c))),
        "exactcircle" => {
            make_exact_geo_circle(pm, f[0], f[1], f[2], f[3]).map(|c| Some(circle_handle(c)))
        }
        "solid" => {
            make_xyz_solid(pm, f[0], f[1], f[2], f[3], f[4], f[5]).map(|s| Some(Handle::solid(s)))
        }
        "bbox" => make_geo_bbox(pm, f[0], f[1], f[2], f[3]).map(|b| Some(bbox_handle(b))),
        // `GeoAreaFactory` returns the same objects behind `GeoArea`; they
        // are downcast back to what Java's `instanceof` checks see.
        "areaxyz" => {
            make_geo_area(pm, f[0], f[1], f[2], f[3], f[4], f[5]).map(|a| Some(area_handle(a)))
        }
        "areall" => make_geo_area_lat_lon(pm, f[0], f[1], f[2], f[3]).map(|a| Some(area_handle(a))),
        "point" => {
            GeoDegeneratePoint::new(pm, f[0], f[1]).map(|p| Some(Handle::point(Arc::new(p))))
        }
        "polygon" => {
            let mut at = 0;
            read_description(args, &mut at, pm)
                .and_then(|desc| make_geo_polygon_from_description(pm, &desc))
                .map(|p| p.map(|p| Handle::area_shape(p)))
        }
        "convex" | "concave" => {
            let mut at = 0;
            read_points(args, &mut at, pm)
                .and_then(|pts| {
                    if kind == "convex" {
                        make_geo_convex_polygon(pm, pts)
                    } else {
                        make_geo_concave_polygon(pm, pts)
                    }
                })
                .map(|p| Some(Handle::area_shape(p)))
        }
        "largepolygon" => {
            let count: usize = args[0].parse().unwrap();
            let mut at = 1;
            (0..count)
                .map(|_| read_description(args, &mut at, pm))
                .collect::<Result<Vec<_>>>()
                .and_then(|ds| make_large_geo_polygon(pm, &ds))
                .map(|p| Some(Handle::area_shape(p)))
        }
        "s2" => {
            let mut at = 0;
            read_points(args, &mut at, pm)
                .and_then(|g| {
                    let mut g = g.into_iter();
                    let mut next = || g.next().unwrap();
                    make_geo_s2_shape(pm, next(), next(), next(), next())
                })
                .map(|p| Some(Handle::area_shape(p)))
        }
        _ => return None,
    })
}

#[derive(Default)]
struct Run {
    failures: Vec<String>,
    checked: HashMap<&'static str, usize>,
    skipped_kinds: HashMap<String, usize>,
}

impl Run {
    fn check(&mut self, what: &'static str, ok: bool, detail: impl FnOnce() -> String) {
        *self.checked.entry(what).or_default() += 1;
        if !ok && self.failures.len() < 60 {
            self.failures.push(format!("{what}: {}", detail()));
        } else if !ok {
            self.failures.push(String::new());
        }
    }
}

fn xyz_bounds_text(b: &XYZBounds) -> String {
    [
        b.minimum_x(),
        b.maximum_x(),
        b.minimum_y(),
        b.maximum_y(),
        b.minimum_z(),
        b.maximum_z(),
    ]
    .iter()
    .map(|v| opt(*v))
    .collect::<Vec<_>>()
    .join("\t")
}

/// Compares six recorded optional doubles with computed bounds.
fn same_bounds(want: &[&str], b: &XYZBounds) -> bool {
    let got = [
        b.minimum_x(),
        b.maximum_x(),
        b.minimum_y(),
        b.maximum_y(),
        b.minimum_z(),
        b.maximum_z(),
    ];
    want.len() == 6
        && want.iter().zip(got).all(|(w, g)| match (*w, g) {
            ("null", None) => true,
            (w, Some(g)) if w != "null" => same(d(w), g),
            _ => false,
        })
}

#[test]
fn geo3d_matches_lucene() {
    let text = std::fs::read_to_string(root().join("shapes.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenGeo3d");
    let mut run = Run::default();
    let mut pms: HashMap<String, Arc<PlanetModel>> = HashMap::new();
    let mut shapes: HashMap<String, Handle> = HashMap::new();
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        match a[0] {
            "PM" => {
                let pm = Arc::new(PlanetModel::new(d(a[2]), d(a[3])));
                pms.insert(a[1].to_string(), pm);
            }
            "PMV" => {
                let pm = &pms[a[1]];
                let got = format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}",
                    h(pm.max_value),
                    h(pm.decode),
                    pm.min_encoded_value,
                    pm.max_encoded_value,
                    h(pm.minimum_pole_distance),
                    h(pm.mean_radius()),
                    h(pm.scaled_flattening)
                );
                let want = a[2..].join("\t");
                run.check("PMV", got == want, || format!("{line}\n  rust: {got}"));
            }
            "ENC" => {
                let pm = &pms[a[1]];
                let got = match pm.encode_value(d(a[2])) {
                    Ok(e) => format!("OK\t{e}\t{}", h(pm.decode_value(e))),
                    Err(e) => err_text(&e),
                };
                let want = a[3..].join("\t");
                run.check("ENC", got == want, || format!("{line}\n  rust: {got}"));
            }
            "DVE" => {
                let pm = &pms[a[1]];
                let enc = pm.doc_value_encoder();
                let got = match enc.encode_point_xyz(d(a[2]), d(a[3]), d(a[4])) {
                    Ok(dv) => {
                        let back = enc.decode_point(dv);
                        format!("OK\t{dv}\t{}\t{}\t{}", h(back.x), h(back.y), h(back.z))
                    }
                    Err(e) => err_text(&e),
                };
                let want = a[5..].join("\t");
                run.check("DVE", got == want, || format!("{line}\n  rust: {got}"));
            }
            "SD" => {
                let pm = &pms[a[1]];
                let pa = GeoPoint::from_lat_lon(pm, d(a[2]), d(a[3])).unwrap();
                let pb = GeoPoint::from_lat_lon(pm, d(a[4]), d(a[5])).unwrap();
                let dist = pm.surface_distance(&pa, &pb);
                run.check("SD.distance", same(dist, d(a[6])), || {
                    format!("{line}\n  rust: {}", h(dist))
                });
                let bear = match pm.surface_point_on_bearing(&pa, d(a[7]), d(a[8])) {
                    Ok(c) => format!("OK\t{}\t{}\t{}", h(c.x), h(c.y), h(c.z)),
                    Err(e) => err_text(&e),
                };
                let want = a[9..12].join("\t");
                run.check(
                    "SD.bearing",
                    bear == want
                        || (a[9] == "OK"
                            && bear.starts_with("OK")
                            && bear
                                .split('\t')
                                .skip(1)
                                .zip(a[10..13].iter())
                                .all(|(g, w)| same(d(g), d(w)))),
                    || format!("{line}\n  rust: {bear}"),
                );
                let bis = pm.bisection(&pa, &pb).map_or("null".to_string(), |p| {
                    format!("{},{},{}", h(p.x), h(p.y), h(p.z))
                });
                run.check("SD.bisection", bis == *a.last().unwrap(), || {
                    format!("{line}\n  rust: {bis}")
                });
            }
            "SH" => {
                let pm = &pms[a[2]];
                let args: Vec<&str> = a[4].split(' ').collect();
                let Some(built) = build(a[3], &args, pm) else {
                    *run.skipped_kinds.entry(a[3].to_string()).or_default() += 1;
                    continue;
                };
                let want = a[6..].join("\t");
                match built {
                    Ok(Some(handle)) => {
                        let mut bytes = Vec::new();
                        let ser = write_planet_object(&mut bytes, &*handle.planet);
                        let code = handle
                            .planet
                            .class_code()
                            .and_then(class_name)
                            .unwrap_or("?");
                        let got = match ser {
                            Ok(()) => format!("OK\t{code}\t{}", hex(&bytes)),
                            Err(e) => err_text(&e),
                        };
                        run.check("SH", got == want, || format!("{line}\n  rust: {got}"));
                        // Rust reads Lucene's bytes back to the same bytes.
                        if let Some(java_hex) = a.get(8) {
                            let java = unhex(java_hex);
                            let again = read_planet_object(&mut Input::new(&java)).and_then(|o| {
                                let mut out = Vec::new();
                                write_planet_object(&mut out, &*o.as_planet_object().unwrap())?;
                                Ok(out)
                            });
                            run.check("SH.read", again.as_ref().is_ok_and(|b| *b == java), || {
                                format!("{line}\n  rust read: {:?}", again.map(|b| hex(&b)))
                            });
                        }
                        shapes.insert(a[1].to_string(), handle);
                    }
                    Ok(None) => run.check("SH", want == "NULL", || format!("{line}\n  rust: NULL")),
                    Err(e) => {
                        let got = err_text(&e);
                        run.check("SH", same_err(&got, &want), || {
                            format!("{line}\n  rust: {got}")
                        });
                    }
                }
            }
            _ => {
                // A record about a shape: skip it when the shape was skipped.
                let Some(handle) = shapes.get(a[1]) else {
                    continue;
                };
                check_shape_record(&mut run, handle, &a, line, &shapes, &pms);
            }
        }
    }
    let mut counts: Vec<_> = run.checked.iter().collect();
    counts.sort();
    eprintln!("checked: {counts:?}");
    eprintln!("skipped kinds: {:?}", run.skipped_kinds);
    // A shape the port cannot build would skip its records silently, and an
    // empty or truncated corpus would check nothing: neither may pass.
    assert!(
        run.skipped_kinds.is_empty(),
        "shapes skipped: {:?}",
        run.skipped_kinds
    );
    for kind in [
        "PMV",
        "ENC",
        "DVE",
        "SD.distance",
        "SH",
        "SH.read",
        "XB",
        "LB",
        "EP",
        "RS",
        "EX",
        "DB",
        "Q.within",
        "Q.outside",
        "Q.distance",
        "Q.path",
        "Q.kinds",
        "REL",
        "XREL",
    ] {
        assert!(
            run.checked.get(kind).copied().unwrap_or(0) > 0,
            "no {kind} record checked"
        );
    }
    let n = run.failures.len();
    let shown: Vec<_> = run
        .failures
        .iter()
        .filter(|f| !f.is_empty())
        .take(40)
        .cloned()
        .collect();
    assert!(n == 0, "{n} mismatches:\n{}", shown.join("\n"));
}

fn check_shape_record(
    run: &mut Run,
    handle: &Handle,
    a: &[&str],
    line: &str,
    shapes: &HashMap<String, Handle>,
    pms: &HashMap<String, Arc<PlanetModel>>,
) {
    let _ = pms;
    match a[0] {
        "XB" => {
            let shape = handle.shape.as_ref().expect("XB on a shape");
            let mut b = XYZBounds::new();
            shape.get_bounds(&mut b);
            let got = format!("OK\t{}", xyz_bounds_text(&b));
            run.check("XB", a[2] == "OK" && same_bounds(&a[3..], &b), || {
                format!("{line}\n  rust: {got}")
            });
        }
        "LB" => {
            let shape = handle.shape.as_ref().expect("LB on a shape");
            let mut b = LatLonBounds::new();
            shape.get_bounds(&mut b);
            let flags = format!(
                "{}{}{}",
                u8::from(b.check_no_top_latitude_bound()),
                u8::from(b.check_no_bottom_latitude_bound()),
                u8::from(b.check_no_longitude_bound())
            );
            let vals = [
                b.max_latitude(),
                b.min_latitude(),
                b.left_longitude(),
                b.right_longitude(),
            ];
            let ok = a[2] == "OK"
                && a[3] == flags
                && a[4..8].iter().zip(vals).all(|(w, g)| match (*w, g) {
                    ("null", None) => true,
                    (w, Some(g)) if w != "null" => same(d(w), g),
                    _ => false,
                });
            run.check("LB", ok, || {
                format!(
                    "{line}\n  rust: OK\t{flags}\t{}",
                    vals.iter().map(|v| opt(*v)).collect::<Vec<_>>().join("\t")
                )
            });
        }
        "EP" => {
            let shape = handle.shape.as_ref().expect("EP on a shape");
            let eps = shape.edge_points();
            let mut got = eps.len().to_string();
            for e in eps.iter() {
                got.push_str(&format!(" {} {} {}", h(e.x), h(e.y), h(e.z)));
            }
            run.check("EP", got == a[2], || format!("{line}\n  rust: {got}"));
        }
        "RS" => {
            let s = handle.sizeable.as_ref().expect("RS on a sizeable");
            let c = s.center();
            let ok = same_str(a[2], s.radius())
                && same_str(a[3], c.x)
                && same_str(a[4], c.y)
                && same_str(a[5], c.z);
            run.check("RS", ok, || {
                format!(
                    "{line}\n  rust: {}\t{}\t{}\t{}",
                    h(s.radius()),
                    h(c.x),
                    h(c.y),
                    h(c.z)
                )
            });
        }
        "EX" => {
            let b = handle.bbox.as_ref().expect("EX on a bbox");
            let got = match b.expand(d(a[2])) {
                Ok(e) => {
                    let mut bytes = Vec::new();
                    match write_planet_object(&mut bytes, &*e) {
                        Ok(()) => format!(
                            "OK\t{}\t{}",
                            e.class_code().and_then(class_name).unwrap_or("?"),
                            hex(&bytes)
                        ),
                        Err(e) => err_text(&e),
                    }
                }
                Err(e) => err_text(&e),
            };
            run.check("EX", got == a[3..].join("\t"), || {
                format!("{line}\n  rust: {got}")
            });
        }
        "DB" => {
            let s = handle.distance.as_ref().expect("DB on a distance shape");
            let mut b = XYZBounds::new();
            let got = match s.get_distance_bounds(&mut b, style(a[2]), d(a[3])) {
                Ok(()) => format!("OK\t{}", xyz_bounds_text(&b)),
                Err(e) => err_text(&e),
            };
            let ok = if a[4] == "OK" {
                got.starts_with("OK") && same_bounds(&a[5..], &b)
            } else {
                got == a[4..].join("\t")
            };
            run.check("DB", ok, || format!("{line}\n  rust: {got}"));
        }
        "Q" => {
            // `Q id x y z within style [O outside] [D distance delta]
            // [P nearest center]`: a double or `ERR class message` each.
            let (x, y, z) = (d(a[2]), d(a[3]), d(a[4]));
            let within = match catch(|| handle.membership.is_within_xyz(x, y, z)) {
                Ok(true) => "1".to_string(),
                Ok(false) => "0".to_string(),
                Err(e) => err_text(&e),
            };
            let n = if a[5] == "ERR" { 3 } else { 1 };
            run.check("Q.within", within == a[5..5 + n].join("\t"), || {
                format!("{line}\n  rust: {within}")
            });
            let st = style(a[5 + n]);
            let mut i = 6 + n;
            let mut tags = String::new();
            while i < a.len() {
                let tag = a[i];
                tags.push_str(tag);
                i += 1;
                let (what, got): (&'static str, Vec<Result<f64>>) = match tag {
                    "O" => {
                        let s = handle.outside.as_ref().expect("O on a membership shape");
                        (
                            "Q.outside",
                            vec![catch(|| s.compute_outside_distance(st, x, y, z))],
                        )
                    }
                    "D" => {
                        let s = handle.distance.as_ref().expect("D on a distance shape");
                        (
                            "Q.distance",
                            vec![
                                catch(|| s.compute_distance(st, x, y, z)),
                                catch(|| s.compute_delta_distance(st, x, y, z)),
                            ],
                        )
                    }
                    "P" => {
                        let s = handle.path.as_ref().expect("P on a path");
                        (
                            "Q.path",
                            vec![
                                catch(|| s.compute_nearest_distance(st, x, y, z)),
                                catch(|| s.compute_path_center_distance(st, x, y, z)),
                            ],
                        )
                    }
                    other => panic!("unknown Q tag {other}: {line}"),
                };
                let mut ok = true;
                for v in &got {
                    let (same, used) = same_num(&a[i..], v);
                    ok &= same;
                    i += used;
                }
                run.check(what, ok, || {
                    let nums: Vec<String> = got.iter().cloned().map(num).collect();
                    format!("{line}\n  rust: {}", nums.join("\t"))
                });
            }
            // Java answers a distance kind exactly when the shape is one; so
            // must the port's handle.
            let expected = format!(
                "{}{}{}",
                if handle.outside.is_some() { "O" } else { "" },
                if handle.distance.is_some() { "D" } else { "" },
                if handle.path.is_some() { "P" } else { "" }
            );
            run.check("Q.kinds", tags == expected, || {
                format!("{line}\n  rust kinds: {expected}")
            });
        }
        "REL" => {
            let Some(other) = shapes.get(a[2]) else {
                return;
            };
            let area = handle.area.as_ref().expect("REL on an area");
            let shape = other.shape.as_ref().expect("REL against a shape");
            let got = match catch(|| area.get_relationship(&**shape)).and_then(|r| r) {
                Ok(r) => rel_name(r).to_string(),
                Err(e) => err_text(&e),
            };
            run.check("REL", got == a[3..].join("\t"), || {
                format!("{line}\n  rust: {got}")
            });
        }
        "XREL" => {
            let shape = handle.shape.as_ref().expect("XREL against a shape");
            let pm = shape.planet_model();
            let v: Vec<f64> = a[2].split(' ').map(d).collect();
            let got = match make_xyz_solid(pm, v[0], v[1], v[2], v[3], v[4], v[5]) {
                Ok(solid) => {
                    let name = solid.class_code().and_then(class_name).unwrap_or("?");
                    match catch(|| solid.get_relationship(&**shape)).and_then(|r| r) {
                        Ok(r) => format!("{name}\t{}", rel_name(r)),
                        Err(e) => err_text(&e),
                    }
                }
                Err(e) => err_text(&e),
            };
            run.check("XREL", got == a[3..].join("\t"), || {
                format!("{line}\n  rust: {got}")
            });
        }
        other => panic!("unknown record {other}"),
    }
}
