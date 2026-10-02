//! The Spatial4j subset, Lucene's Geo3D bridge to it and the S2 cell ids,
//! differentially against Spatial4j 0.8, Lucene 10.5.0 and
//! s2-geometry-library-java 1.0.0: `fixtures/src/GenSpatial4j.java` built
//! seeded random shapes in eight spatial contexts (geodetic with each
//! distance formula, normalising longitudes, planar unbounded and bounded,
//! Geo3D on a sphere and on WGS84) and recorded their `toString`s, bounding
//! boxes, centers, areas, buffers, pairwise relations and equality,
//! distances, binary encodings, WKT parses (results, error messages and
//! offsets), `DistanceUtils`, geohashes and S2 cell ids and vertices. This
//! test makes the same calls and compares the formatted results -- bit for
//! bit for doubles (a NaN's sign aside, which differs between x86-64 and
//! arm64), exactly for everything else.

use std::collections::BTreeMap;
use std::sync::Arc;

use lucene_util::s2::{projections, S2Cell, S2CellId, S2LatLng};
use lucene_util::spatial4j::binary_codec::DataInput;
use lucene_util::spatial4j::geohash;
use lucene_util::spatial4j::{
    DistanceUtils, Error, Point, Shape, SpatialContext, SpatialContextFactory,
};

/// `GenSpatial4j.CTX_ARGS`.
const CTX_ARGS: &[&[(&str, &str)]] = &[
    &[],
    &[("distCalculator", "lawOfCosines")],
    &[("distCalculator", "vincentySphere")],
    &[("normWrapLongitude", "true")],
    &[("geo", "false")],
    &[
        ("geo", "false"),
        ("worldBounds", "ENVELOPE(-1000, 1000, 1000, -1000)"),
    ],
    &[(
        "spatialContextFactory",
        "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
    )],
    &[
        (
            "spatialContextFactory",
            "org.apache.lucene.spatial.spatial4j.Geo3dSpatialContextFactory",
        ),
        ("planetModel", "wgs84"),
    ],
];

fn contexts() -> Vec<Arc<SpatialContext>> {
    CTX_ARGS
        .iter()
        .map(|args| {
            let m: BTreeMap<String, String> = args
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            SpatialContextFactory::make_spatial_context(&m).expect("a valid context")
        })
        .collect()
}

fn root() -> std::path::PathBuf {
    match std::env::var_os("SPATIAL4J_FIXTURES") {
        Some(dir) => dir.into(),
        None => {
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/spatial4j")
        }
    }
}

fn h(v: f64) -> String {
    format!("{:x}", v.to_bits())
}

fn d(s: &str) -> f64 {
    f64::from_bits(u64::from_str_radix(s, 16).unwrap())
}

fn esc(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\t', "\\t")
        .replace('\n', "\\n")
}

/// `GenSpatial4j.err(e)`. A `NullPointerException`'s message is the JVM's
/// own (helpful NPE text), so only its class is recorded.
fn err(e: &Error) -> String {
    let mut s = format!("ERR {} {}", e.java_class(), esc(&e.to_string()));
    if let Error::Parse { offset, .. } = e {
        s.push_str(&format!(" @{offset}"));
    }
    s
}

fn safe<T: ToString>(r: Result<T, Error>) -> String {
    match r {
        Ok(v) => v.to_string(),
        Err(e) => err(&e),
    }
}

fn bbox(s: &dyn Shape) -> Result<String, Error> {
    let r = s.bounding_box()?;
    Ok(format!(
        "{},{},{},{}",
        h(r.min_x()),
        h(r.max_x()),
        h(r.min_y()),
        h(r.max_y())
    ))
}

fn pt(p: &dyn Point) -> String {
    format!("{},{}", h(p.x()), h(p.y()))
}

/// Rebuilds a `GenSpatial4j.parse` spec.
fn parse(ctx: &Arc<SpatialContext>, t: &[&str], pos: &mut usize) -> Result<Arc<dyn Shape>, Error> {
    let k = t[*pos];
    *pos += 1;
    let mut next = || {
        let v = d(t[*pos]);
        *pos += 1;
        v
    };
    Ok(match k {
        "P" => {
            let (x, y) = (next(), next());
            ctx.point_xy(x, y)?
        }
        "R" => {
            let (a, b, c, e) = (next(), next(), next(), next());
            ctx.rect(a, b, c, e)?
        }
        "C" => {
            let (x, y, r) = (next(), next(), next());
            ctx.circle(x, y, r)?
        }
        "L" => {
            let buf = next();
            let n: usize = t[*pos].parse().unwrap();
            *pos += 1;
            let mut pts = Vec::new();
            for _ in 0..n {
                let x = d(t[*pos]);
                let y = d(t[*pos + 1]);
                *pos += 2;
                pts.push(ctx.point_xy(x, y)?);
            }
            ctx.line_string(&pts, buf)?
        }
        "M" => {
            let n: usize = t[*pos].parse().unwrap();
            *pos += 1;
            let mut shapes = Vec::new();
            for _ in 0..n {
                shapes.push(parse(ctx, t, pos)?);
            }
            Arc::new(ctx.collection(shapes)?)
        }
        other => panic!("unknown spec {other}"),
    })
}

fn shape(ctx: &Arc<SpatialContext>, spec: &str) -> Result<Arc<dyn Shape>, Error> {
    let t: Vec<&str> = spec.split(' ').collect();
    parse(ctx, &t, &mut 0)
}

/// The record's result as this port computes it.
fn compute(ctxs: &[Arc<SpatialContext>], op: &str, f: &[&str]) -> String {
    let ctx_of = |s: &str| -> (usize, &Arc<SpatialContext>) {
        let c: usize = s.parse().unwrap();
        (c, &ctxs[c])
    };
    match op {
        "shape" => {
            let (c, ctx) = ctx_of(f[0]);
            let bd = d(f[2]);
            safe(shape(ctx, f[1]).map(|sh| {
                let mut s = String::new();
                s.push_str(&if c >= 6 {
                    "-".into()
                } else {
                    esc(&sh.to_string())
                });
                s.push_str(&format!(" | {}", safe(bbox(&*sh))));
                s.push_str(&format!(" | {}", safe(sh.center().map(|p| pt(&*p)))));
                s.push_str(&format!(" | {} {}", sh.has_area(), sh.is_empty()));
                s.push_str(&format!(" | {}", safe(sh.area(Some(ctx)).map(h))));
                s.push_str(&format!(" | {}", safe(sh.area(None).map(h))));
                let buffered = sh.buffered(bd, ctx).and_then(|b| {
                    let t = if c >= 6 {
                        "-".into()
                    } else {
                        esc(&b.to_string())
                    };
                    Ok(format!("{t} {}", bbox(&*b)?))
                });
                s.push_str(&format!(" | {}", safe(buffered)));
                s
            }))
        }
        "rel" => {
            let (_, ctx) = ctx_of(f[0]);
            let r = shape(ctx, f[1]).and_then(|a| {
                let b = shape(ctx, f[2])?;
                Ok(format!("{} {}", safe(a.relate(&*b)), a.equals(&*b)))
            });
            safe(r)
        }
        "dist" => {
            let (_, ctx) = ctx_of(f[0]);
            let v: Vec<f64> = f[1].split(' ').map(d).collect();
            let (x1, y1, x2, y2, dd, bearing) = (v[0], v[1], v[2], v[3], v[4], v[5]);
            let r = ctx.point_xy(x1, y1).and_then(|p| {
                let q = ctx.point_xy(x2, y2)?;
                let calc = ctx.dist_calc();
                Ok(format!(
                    "{} {} {} {} {} {}",
                    safe(calc.distance(&*p, &*q).map(h)),
                    safe(calc.distance_xy(&*p, x2, y2).map(h)),
                    safe(calc.within(&*p, x2, y2, dd)),
                    safe(calc.point_on_bearing(&p, dd, bearing, ctx).map(|r| pt(&*r))),
                    safe(
                        calc.calc_box_by_dist_from_pt(&p, dd, ctx)
                            .and_then(|r| bbox(&*r))
                    ),
                    safe(
                        calc.calc_box_by_dist_from_pt_y_horiz_axis_deg(&*p, dd, ctx)
                            .map(h)
                    ),
                ))
            });
            safe(r)
        }
        "codec" => {
            let (_, ctx) = ctx_of(f[0]);
            let r = shape(ctx, f[1]).and_then(|sh| {
                let mut out = Vec::new();
                ctx.binary_codec().write_shape(&mut out, &*sh)?;
                let hex: String = out.iter().map(|b| format!("{b:02x}")).collect();
                let back = ctx
                    .binary_codec()
                    .read_shape(ctx, &mut DataInput::new(&out))?;
                Ok(format!("{hex} {} {}", bbox(&*back)?, back.equals(&*sh)))
            });
            safe(r)
        }
        "wkt" => {
            let (c, ctx) = ctx_of(f[0]);
            let wkt = f[1]
                .replace("\\t", "\t")
                .replace("\\n", "\n")
                .replace("\\\\", "\\");
            let r = ctx.read_shape_from_wkt(&wkt).map(|s| {
                format!(
                    "{} | {}",
                    if c >= 6 {
                        "-".into()
                    } else {
                        esc(&s.to_string())
                    },
                    safe(bbox(&*s))
                )
            });
            safe(r)
        }
        "du" => {
            let v: Vec<f64> = f[0].split(' ').map(d).collect();
            let (a, b, dd, la1, lo1, la2, lo2) = (v[0], v[1], v[2], v[3], v[4], v[5], v[6]);
            [
                DistanceUtils::norm_lon_deg(a),
                DistanceUtils::norm_lat_deg(b),
                DistanceUtils::calc_box_by_dist_from_pt_delta_lon_deg(b, a, dd),
                DistanceUtils::calc_box_by_dist_from_pt_lat_horiz_axis_deg(b, a, dd),
                DistanceUtils::calc_lon_degrees_at_lat(b, dd),
                DistanceUtils::dist_haversine_rad(la1, lo1, la2, lo2),
                DistanceUtils::dist_law_of_cosines_rad(la1, lo1, la2, lo2),
                DistanceUtils::dist_vincenty_rad(la1, lo1, la2, lo2),
                DistanceUtils::dist2_degrees(dd, DistanceUtils::EARTH_MEAN_RADIUS_KM),
                DistanceUtils::degrees2_dist(dd, DistanceUtils::EARTH_MEAN_RADIUS_KM),
                DistanceUtils::to_radians(a),
                DistanceUtils::to_degrees(la1),
            ]
            .iter()
            .map(|&x| h(x))
            .collect::<Vec<_>>()
            .join(" ")
        }
        "gh" => {
            let v: Vec<&str> = f[0].split(' ').collect();
            let (la, lo, prec) = (d(v[0]), d(v[1]), v[2].parse::<usize>().unwrap());
            let hash = geohash::encode_lat_lon(la, lo, prec);
            let r = geohash::decode_boundary(&hash, &ctxs[0]).unwrap();
            let p = geohash::decode(&hash.to_uppercase(), &ctxs[0]).unwrap();
            format!(
                "{hash} {},{},{},{} {} {}",
                h(r.min_x()),
                h(r.max_x()),
                h(r.min_y()),
                h(r.max_y()),
                pt(&*p),
                geohash::sub_geohashes(&hash[..prec - 1]).join(",")
            )
        }
        "ghlen" => {
            let v: Vec<f64> = f[0].split(' ').map(d).collect();
            geohash::lookup_hash_len_for_width_height(v[0], v[1]).to_string()
        }
        "ghsize" => {
            let s = geohash::lookup_degrees_size_for_hash_len(f[0].parse().unwrap());
            format!("{} {}", h(s[0]), h(s[1]))
        }
        "s2" => {
            let v: Vec<&str> = f[0].split(' ').collect();
            let (la, lo, level) = (d(v[0]), d(v[1]), v[2].parse::<i32>().unwrap());
            let leaf = S2CellId::from_lat_lng(&S2LatLng::from_degrees(la, lo));
            let id = leaf.parent(level);
            let mut sb = format!(
                "{:x} {:x} {} {} {} {} {} {} {}",
                leaf.id() as u64,
                id.id() as u64,
                id.level(),
                id.face(),
                id.to_token(),
                id.to_string().replace(' ', "_"),
                id.is_leaf(),
                id.is_face(),
                id.is_valid()
            );
            for l in 1..=level {
                sb.push_str(if l == 1 { " " } else { "," });
                sb.push_str(&id.child_position(l).to_string());
            }
            if level < lucene_util::s2::MAX_LEVEL {
                let child = id.child_begin(level + 1);
                let cmp = |a: &S2CellId, b: &S2CellId| a.cmp(b) as i8;
                sb.push_str(&format!(
                    " {:x} {:x} {} {} {} {}",
                    child.id() as u64,
                    child.next().id() as u64,
                    id.contains(&child),
                    child.contains(&id),
                    cmp(&id, &child),
                    cmp(&child, &id.next())
                ));
            }
            let cell = S2Cell::new(id);
            for k in 0..4 {
                let p = cell.vertex_raw(k);
                sb.push_str(&format!(" {},{},{}", h(p.x), h(p.y), h(p.z)));
            }
            sb
        }
        "s2face" => format!(
            "{:x}",
            S2CellId::from_face_pos_level(f[0].parse().unwrap(), 0, 0).id() as u64
        ),
        "s2min" => projections::MAX_WIDTH.get_min_level(d(f[0])).to_string(),
        "s2val" => h(projections::MAX_WIDTH.get_value(f[0].parse().unwrap())),
        other => panic!("unknown op {other}"),
    }
}

/// Replaces every 16-digit hex token that is a NaN's bits with `NaN`, and
/// a `NullPointerException`'s message with nothing.
fn normalise(text: &str) -> String {
    let text = match text.find("ERR java.lang.NullPointerException") {
        Some(i) => {
            let end = text[i..].find(" | ").map_or(text.len(), |e| i + e);
            format!(
                "{}ERR java.lang.NullPointerException{}",
                &text[..i],
                &text[end..]
            )
        }
        None => text.to_string(),
    };
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let run = bytes[i..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count();
        if run == 0 {
            let skip = bytes[i..]
                .iter()
                .position(u8::is_ascii_hexdigit)
                .unwrap_or(bytes.len() - i);
            out.push_str(&text[i..i + skip]);
            i += skip;
            continue;
        }
        let token = &text[i..i + run];
        let is_nan = run == 16
            && u64::from_str_radix(token, 16).is_ok_and(|bits| f64::from_bits(bits).is_nan());
        out.push_str(if is_nan { "NaN" } else { token });
        i += run;
    }
    out
}

#[test]
fn normalise_ignores_nan_sign_and_npe_text() {
    assert_eq!(
        normalise("fff8000000000000,1"),
        normalise("7ff8000000000000,1")
    );
    assert_ne!(normalise("3ff0000000000000"), normalise("bff0000000000000"));
    assert_eq!(
        normalise("x | ERR java.lang.NullPointerException because y | z"),
        "x | ERR java.lang.NullPointerException | z"
    );
}

#[test]
fn spatial4j_matches_java() {
    let text = std::fs::read_to_string(root().join("spatial4j.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenSpatial4j");
    let ctxs = contexts();
    let mut failures = Vec::new();
    let mut checked = 0;
    for line in text.lines() {
        let (lhs, expected) = line.split_once("\t=>\t").expect("a record");
        let mut fields: Vec<&str> = lhs.split('\t').collect();
        let op = fields.remove(0);
        let actual = compute(&ctxs, op, &fields);
        checked += 1;
        if normalise(&actual) != normalise(expected) {
            failures.push(format!("{lhs}\n   java: {expected}\n   rust: {actual}"));
        }
    }
    assert!(checked > 10_000, "only {checked} records");
    assert!(
        failures.is_empty(),
        "{} of {checked} records differ:\n{}",
        failures.len(),
        failures
            .iter()
            .take(15)
            .cloned()
            .collect::<Vec<_>>()
            .join("\n")
    );
}
