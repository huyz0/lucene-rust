//! geo3d's geometric primitives differentially against Lucene 10.5.0:
//! `fixtures/src/GenGeo3dMath.java` called `Vector`, `GeoPoint`, `Plane`,
//! `SidedPlane`, `PlanetModel`, `XYZBounds`, `LatLonBounds` and
//! `DistanceStyle` directly on seeded random (and near-degenerate) inputs;
//! this test makes the same calls and compares the formatted results -- bit
//! for bit for doubles, exactly for booleans, hash codes, `toString`s and
//! exception messages.

#![allow(non_snake_case)]

use std::sync::Arc;

use lucene_util::geo::java_double_string;
use lucene_util::spatial3d::membership::Membership;
use lucene_util::spatial3d::plane::Plane;
use lucene_util::spatial3d::*;
use lucene_util::strict_math;

/// The committed corpus, or `GEO3D_FIXTURES` -- a directory a generator run
/// with another `-Dgeo3d.seed` wrote, for a wider one-off sweep.
fn root() -> std::path::PathBuf {
    match std::env::var_os("GEO3D_FIXTURES") {
        Some(dir) => dir.into(),
        None => std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/geo3d"),
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

fn err(e: &Error) -> String {
    format!("ERR {} {}", e.java_class(), esc(&e.to_string()))
}

fn fv(v: &Vector) -> String {
    format!("{},{},{}", h(v.x), h(v.y), h(v.z))
}

fn fov(v: Option<&Vector>) -> String {
    v.map_or("null".to_string(), fv)
}

fn fp(p: &Plane) -> String {
    format!("{},{},{},{}", h(p.x), h(p.y), h(p.z), h(p.D))
}

fn fop(p: Option<&Plane>) -> String {
    p.map_or("null".to_string(), fp)
}

fn fsp(p: &SidedPlane) -> String {
    format!("{},{}", fp(p), h(p.sig_num))
}

fn fpts(pts: Option<&[GeoPoint]>) -> String {
    match pts {
        None => "null".into(),
        Some(pts) => format!(
            "[{}]",
            pts.iter().map(|p| fv(p)).collect::<Vec<_>>().join(";")
        ),
    }
}

fn opt(v: Option<f64>) -> String {
    v.map_or("null".to_string(), h)
}

fn fxb(b: &XYZBounds) -> String {
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
    .join(",")
}

fn flb(b: &LatLonBounds) -> String {
    format!(
        "{},{},{},{},{},{},{}",
        b.check_no_longitude_bound(),
        b.check_no_top_latitude_bound(),
        b.check_no_bottom_latitude_bound(),
        opt(b.max_latitude()),
        opt(b.min_latitude()),
        opt(b.left_longitude()),
        opt(b.right_longitude())
    )
}

/// `Result` -> the generator's text for it.
fn r(v: Result<String>) -> String {
    v.unwrap_or_else(|e| err(&e))
}

/// The tokens of one record's inputs, read in order.
struct In<'a> {
    t: Vec<&'a str>,
    at: usize,
}

impl In<'_> {
    fn f(&mut self) -> f64 {
        self.at += 1;
        d(self.t[self.at - 1])
    }

    fn i(&mut self) -> usize {
        self.at += 1;
        self.t[self.at - 1].parse().unwrap()
    }

    fn v(&mut self) -> Vector {
        Vector::new(self.f(), self.f(), self.f())
    }

    fn g(&mut self) -> GeoPoint {
        GeoPoint::new(self.f(), self.f(), self.f())
    }

    fn plane(&mut self) -> Plane {
        Plane::new(self.f(), self.f(), self.f(), self.f())
    }

    /// A bounds spec: a count (as a double), then nine doubles per
    /// `SidedPlane(p, A, B)`.
    fn bounds(&mut self) -> Result<Vec<SidedPlane>> {
        let k = self.f() as usize;
        let specs: Vec<[Vector; 3]> = (0..k).map(|_| [self.v(), self.v(), self.v()]).collect();
        specs
            .iter()
            .map(|[p, a, b]| SidedPlane::from_vectors(p, a, b))
            .collect()
    }
}

fn members(b: &[SidedPlane]) -> Vec<&dyn Membership> {
    b.iter().map(|p| p as &dyn Membership).collect()
}

const STYLES: [DistanceStyle; 5] = [
    DistanceStyle::Arc,
    DistanceStyle::Linear,
    DistanceStyle::LinearSquared,
    DistanceStyle::Normal,
    DistanceStyle::NormalSquared,
];

fn run(op: &str, inp: &mut In<'_>, pms: &[Arc<PlanetModel>]) -> String {
    let pm_at = |inp: &mut In<'_>| pms[inp.i()].clone();
    match op {
        "PM.str" => {
            let pm = pm_at(inp);
            format!(
                "{}|{}|{}",
                pm.java_to_string(),
                pm.java_hash_code(),
                pm.is_sphere()
            )
        }
        "SM.trig" => {
            let (x, y) = (inp.f(), inp.f());
            format!(
                "{},{},{},{}",
                h(strict_math::tan(x)),
                h(strict_math::atan(x)),
                h(strict_math::atan2(y, x)),
                h(strict_math::atan2(x, y))
            )
        }
        "V.normalize" => fov(inp.v().normalize().as_ref()),
        "V.perp" => {
            let (a, b) = (inp.v(), inp.v());
            r(Vector::perpendicular(&a, &b).map(|v| fv(&v)))
        }
        "V.perpxyz" => {
            let (a, b) = (inp.v(), inp.v());
            r(Vector::perpendicular_to(&a, b.x, b.y, b.z).map(|v| fv(&v)))
        }
        "V.cpez" => {
            let (a, b, c) = (inp.v(), inp.v(), inp.v());
            r(Vector::cross_product_evaluate_is_zero(&a, &b, &c).map(|v| v.to_string()))
        }
        "V.dot" => {
            let (a, b) = (inp.v(), inp.v());
            format!(
                "{},{}",
                h(a.dot_product(&b)),
                h(a.dot_product_xyz(b.x, b.y, b.z))
            )
        }
        "V.translate" => {
            let (a, b) = (inp.v(), inp.v());
            fv(&a.translate(b.x, b.y, b.z))
        }
        "V.rot" => {
            let (a, ang) = (inp.v(), inp.f());
            let (s, c) = (strict_math::sin(ang), strict_math::cos(ang));
            [
                a.rotate_xy(ang),
                a.rotate_xz(ang),
                a.rotate_zy(ang),
                a.rotate_xy_sc(s, c),
                a.rotate_xz_sc(s, c),
                a.rotate_zy_sc(s, c),
            ]
            .iter()
            .map(fv)
            .collect::<Vec<_>>()
            .join("|")
        }
        "V.dist" => {
            let (a, b) = (inp.v(), inp.v());
            [
                a.linear_distance_squared(&b),
                a.linear_distance(&b),
                a.normal_distance_squared(&b),
                a.normal_distance(&b),
                a.linear_distance_squared_xyz(b.x, b.y, b.z),
                a.linear_distance_xyz(b.x, b.y, b.z),
                a.normal_distance_squared_xyz(b.x, b.y, b.z),
                a.normal_distance_xyz(b.x, b.y, b.z),
                a.magnitude(),
                Vector::magnitude_of(a.x, a.y, a.z),
            ]
            .iter()
            .map(|v| h(*v))
            .collect::<Vec<_>>()
            .join(",")
        }
        "V.same" => {
            let (a, b) = (inp.v(), inp.v());
            format!(
                "{},{},{},{},{},{},{}",
                a.is_numerically_identical(&b),
                a.is_numerically_identical_xyz(b.x, b.y, b.z),
                a.is_parallel(&b),
                a.is_parallel_xyz(b.x, b.y, b.z),
                a.x == b.x && a.y == b.y && a.z == b.z,
                a.java_hash_code(),
                a
            )
        }
        "G.ctor" => {
            let pm = pm_at(inp);
            let (la, lo) = (inp.f(), inp.f());
            r(GeoPoint::from_lat_lon(&pm, la, lo).map(|p| {
                format!(
                    "{}|{}|{}|{}",
                    fv(&p),
                    h(p.latitude()),
                    h(p.longitude()),
                    h(p.magnitude())
                )
            }))
        }
        "G.mag" => {
            let (m, v) = (inp.f(), inp.v());
            let q = GeoPoint::with_magnitude(m, v.x, v.y, v.z);
            format!(
                "{}|{}|{}|{}",
                fv(&q),
                h(q.latitude()),
                h(q.longitude()),
                h(q.magnitude())
            )
        }
        "G.xyz" => {
            let v = inp.v();
            let q = GeoPoint::new(v.x, v.y, v.z);
            format!(
                "{}|{}|{}|{}|{}",
                h(q.latitude()),
                h(q.longitude()),
                h(q.magnitude()),
                q.java_hash_code(),
                q
            )
        }
        "G.arc" => {
            let (p, v) = (inp.g(), inp.v());
            format!(
                "{},{},{},{},{}",
                h(p.arc_distance_vector(&v)),
                h(p.arc_distance_xyz(v.x, v.y, v.z)),
                p.is_identical(&v),
                p.is_identical_xyz(v.x, v.y, v.z),
                p.is_identical_xyz(p.x, p.y, p.z)
            )
        }
        "G.trig" => {
            let pm = pm_at(inp);
            let (m, la, lo, v) = (inp.f(), inp.f(), inp.f(), inp.v());
            let (sla, slo, cla, clo) = (
                strict_math::sin(la),
                strict_math::sin(lo),
                strict_math::cos(la),
                strict_math::cos(lo),
            );
            r((|| {
                let q = GeoPoint::from_trig_lat_lon(&pm, sla, slo, cla, clo, la, lo)?;
                let rr = GeoPoint::from_trig(&pm, sla, slo, cla, clo);
                let t = GeoPoint::with_lat_lon_xyz(la, lo, v.x, v.y, v.z);
                let s = format!(
                    "{}|{}|{}|{}|{}",
                    fv(&q),
                    fv(&rr),
                    h(rr.latitude()),
                    h(t.latitude()),
                    h(t.magnitude())
                );
                let u = GeoPoint::with_magnitude_lat_lon(m, v.x, v.y, v.z, la, lo)?;
                Ok(format!(
                    "{s}|{}|{}|{}",
                    fv(&u),
                    h(u.longitude()),
                    h(u.magnitude())
                ))
            })())
        }
        "P.ctor" => {
            let _pm = pm_at(inp);
            let (a, b) = (inp.g(), inp.g());
            r((|| {
                Ok(format!(
                    "{}|{}",
                    fp(&Plane::from_vectors(&a, &b)?),
                    fp(&Plane::from_vector_xyz(&a, b.x, b.y, b.z)?)
                ))
            })())
        }
        "P.simple" => {
            let pm = pm_at(inp);
            let (sl, vx, vy, a) = (inp.f(), inp.f(), inp.f(), inp.g());
            let wd = Plane::with_d(&a, sl);
            format!(
                "{}|{}|{}|{}|{}",
                fp(&Plane::horizontal(&pm, sl)),
                fp(&Plane::vertical(vx, vy)),
                fp(&wd),
                fp(&Plane::offset(&wd, true)),
                fp(&Plane::offset(&wd, false))
            )
        }
        "P.center1" => {
            let (pl, a) = (inp.plane(), inp.g());
            r(Plane::construct_perpendicular_center_plane_one_point(&pl, &a).map(|p| fp(&p)))
        }
        "P.center2" => {
            let (a, b) = (inp.g(), inp.g());
            r(Plane::construct_perpendicular_center_plane_two_points(&a, &b).map(|p| fp(&p)))
        }
        "P.norm" => {
            let (a, b, c, D) = (inp.g(), inp.g(), inp.g(), inp.f());
            let pts: [&Vector; 3] = [&a, &b, &c];
            [
                Plane::construct_normalized_z_plane_points(&pts),
                Plane::construct_normalized_y_plane_points(&pts),
                Plane::construct_normalized_x_plane_points(&pts),
                Plane::construct_normalized_z_plane(a.x, a.y),
                Plane::construct_normalized_y_plane(a.x, a.z, D),
                Plane::construct_normalized_x_plane(a.y, a.z, D),
                Plane::construct_normalized_z_plane(a.x * 1e-13, a.y * 1e-13),
                Plane::construct_normalized_y_plane(a.x * 1e-13, a.z * 1e-13, D),
                Plane::construct_normalized_x_plane(a.y * 1e-13, a.z * 1e-13, D),
            ]
            .iter()
            .map(|p| fop(p.as_ref()))
            .collect::<Vec<_>>()
            .join("|")
        }
        "P.eval" => {
            let (pl, a) = (inp.plane(), inp.g());
            format!(
                "{},{},{},{},{}",
                h(pl.evaluate(&a)),
                pl.evaluate_is_zero(&a),
                pl.evaluate_is_zero_xyz(a.x, a.y, a.z),
                fop(pl.normalize().as_ref()),
                pl.java_hash_code()
            )
        }
        "P.dist" => {
            let pm = pm_at(inp);
            let (pl, t) = (inp.plane(), inp.g());
            r(inp.bounds().map(|bs| {
                let bd = members(&bs);
                [
                    pl.arc_distance(&pm, t.x, t.y, t.z, &bd),
                    pl.arc_distance(&pm, t.x, t.y, t.z, &bd),
                    pl.normal_distance(t.x, t.y, t.z, &bd),
                    pl.normal_distance(t.x, t.y, t.z, &bd),
                    pl.normal_distance_squared(t.x, t.y, t.z, &bd),
                    pl.normal_distance_squared(t.x, t.y, t.z, &bd),
                    pl.linear_distance(&pm, t.x, t.y, t.z, &bd),
                    pl.linear_distance(&pm, t.x, t.y, t.z, &bd),
                    pl.linear_distance_squared(&pm, t.x, t.y, t.z, &bd),
                    pl.linear_distance_squared(&pm, t.x, t.y, t.z, &bd),
                ]
                .iter()
                .map(|v| h(*v))
                .collect::<Vec<_>>()
                .join(",")
            }))
        }
        "P.inter" => {
            let pm = pm_at(inp);
            let (pl, q) = (inp.plane(), inp.plane());
            r(inp.bounds().map(|bs| {
                let bd = members(&bs);
                format!(
                    "{}|{}|{}|{}|{}",
                    fpts(pl.find_intersections(&pm, &q, &bd).as_deref()),
                    fpts(pl.find_crossings(&pm, &q, &bd).as_deref()),
                    pl.is_functionally_identical(&q),
                    pl.is_numerically_identical_plane(&q),
                    fov(pl.sample_intersection_point(&pm, &q).as_deref())
                )
            }))
        }
        "P.isect" => {
            let pm = pm_at(inp);
            let (fe, q) = (inp.plane(), inp.plane());
            let (a, b, n1, n2) = (inp.g(), inp.g(), inp.g(), inp.g());
            let bs = inp.bounds();
            let bs2 = inp.bounds();
            r((|| {
                let (bs, bs2) = (bs?, bs2?);
                let (bd, bd2) = (members(&bs), members(&bs2));
                let notable = [a.clone(), b.clone()];
                let notable2 = [n1.clone(), n2.clone()];
                Ok(format!(
                    "{},{},{},{}",
                    fe.intersects(&pm, &q, &notable, &notable2, &bd, &bd2),
                    fe.crosses(&pm, &q, &notable, &notable2, &bd, &bd2),
                    q.intersects(&pm, &fe, &[], &notable, &bd2, &[]),
                    q.crosses(&pm, &fe, &[], &notable, &bd2, &[])
                ))
            })())
        }
        "P.bounds" => {
            let pm = pm_at(inp);
            let (fe, q) = (inp.plane(), inp.plane());
            r(inp.bounds().map(|bs| {
                let bd = members(&bs);
                let mut xb = XYZBounds::new();
                fe.record_bounds_xyz(&pm, &mut xb, &bd);
                let mut xb2 = XYZBounds::new();
                fe.record_bounds_xyz_intersection(&pm, &mut xb2, &q, &bd);
                let mut lb = LatLonBounds::new();
                fe.record_bounds_lat_lon(&pm, &mut lb, &bd);
                let mut lb2 = LatLonBounds::new();
                fe.record_bounds_lat_lon_intersection(&pm, &mut lb2, &q, &bd);
                format!("{}|{}|{}|{}", fxb(&xb), fxb(&xb2), flb(&lb), flb(&lb2))
            }))
        }
        "P.arcpts" => {
            let pm = pm_at(inp);
            let (fe, dist, a, b, c) = (inp.plane(), inp.f(), inp.g(), inp.g(), inp.g());
            r(inp.bounds().and_then(|bs| {
                let bd = members(&bs);
                let mut s = fpts(Some(&fe.find_arc_distance_points(&pm, dist, &a, &bd)?));
                for st in STYLES {
                    s += "|";
                    s += &match st.find_distance_points(&pm, dist, &a, &fe, &bd) {
                        Ok(p) => fpts(Some(&p)),
                        Err(e) => format!("ERR {}", e.java_class()),
                    };
                    s += ",";
                    s += &match (
                        st.find_minimum_arc_distance(&pm, dist),
                        st.find_maximum_arc_distance(&pm, dist),
                    ) {
                        (Ok(x), Ok(y)) => format!("{},{}", h(x), h(y)),
                        (Err(e), _) | (_, Err(e)) => format!("ERR {}", e.java_class()),
                    };
                    s += &format!(
                        ",{},{},{},{},{}",
                        h(st.compute_distance_points(&a, &b)),
                        h(st.compute_distance_to_plane(&pm, &fe, c.x, c.y, c.z, &bd)),
                        h(st.to_aggregation_form(dist)),
                        h(st.from_aggregation_form(dist)),
                        h(st.aggregate_distances(&[dist, dist * 0.5, 1.0]))
                    );
                }
                Ok(s)
            }))
        }
        "P.interp" => {
            let pm = pm_at(inp);
            let (fe, a, b) = (inp.plane(), inp.g(), inp.g());
            let k = inp.i();
            let props: Vec<f64> = (0..k).map(|_| inp.f()).collect();
            r(fe.interpolate(&pm, &a, &b, &props).map(|p| fpts(Some(&p))))
        }
        "P.coplanar" => {
            let (a, b, c) = (inp.g(), inp.g(), inp.g());
            r(Plane::are_points_coplanar(&a, &b, &c).map(|v| v.to_string()))
        }
        "S.ctor" => {
            let pm = pm_at(inp);
            let (p, a, b, c) = (inp.g(), inp.g(), inp.g(), inp.g());
            let _ = c;
            let (sl, vx, vy, vz, D) = (inp.f(), inp.f(), inp.f(), inp.f(), inp.f());
            let v = Vector::new(vx, vy, vz);
            let tries: Vec<Result<SidedPlane>> = vec![
                SidedPlane::from_vectors(&p, &a, &b),
                SidedPlane::from_two_vectors(&a, &b),
                SidedPlane::from_vector_xyz(&p, &a, b.x, b.y, b.z),
                SidedPlane::from_vectors_on_side(&p, true, &a, &b),
                SidedPlane::from_vectors_on_side(&p, false, &a, &b),
                SidedPlane::horizontal(&p, &pm, sl),
                SidedPlane::vertical(&p, vx, vy),
                SidedPlane::from_abcd(&p, vx, vy, vz, D),
                SidedPlane::from_normal(&p, &v, D),
                SidedPlane::from_xyz_normal(p.x, p.y, p.z, &v, D),
                SidedPlane::from_vectors(&p, &a, &b).map(|s| SidedPlane::opposite(&s)),
            ];
            tries
                .iter()
                .map(|t| r(t.as_ref().map(fsp).map_err(Clone::clone)))
                .collect::<Vec<_>>()
                .join("|")
        }
        "S.static" => {
            let _pm = pm_at(inp);
            let (p, a, b, c) = (inp.g(), inp.g(), inp.g(), inp.g());
            let (_sl, vx, vy, vz, _D) = (inp.f(), inp.f(), inp.f(), inp.f(), inp.f());
            let v = Vector::new(vx, vy, vz);
            let fso = |s: Option<SidedPlane>| s.map_or("null".to_string(), |s| fsp(&s));
            let parts = [
                r(
                    SidedPlane::construct_normalized_perpendicular_sided_plane(&p, &v, &a, &b)
                        .map(fso),
                ),
                r(SidedPlane::construct_sided_plane_from_two_points(&p, &a, &b).map(|s| fsp(&s))),
                r(Plane::from_vectors(&a, &b)
                    .and_then(|pl| SidedPlane::construct_sided_plane_from_one_point(&p, &pl, &c))
                    .map(|s| fsp(&s))),
                fso(SidedPlane::construct_normalized_three_point_sided_plane(
                    &p, &a, &b, &c,
                )),
                r(SidedPlane::from_vectors(&p, &a, &b).map(|sp| {
                    format!(
                        "{},{},{},{},{},{},{},{}",
                        sp.is_within(&c),
                        sp.is_within_xyz(c.x, c.y, c.z),
                        sp.strictly_within(&c),
                        sp.strictly_within_xyz(c.x, c.y, c.z),
                        sp.strictly_within(&a),
                        sp.java_hash_code(),
                        true,
                        sp.is_within(&a)
                    )
                })),
            ];
            parts.join("|")
        }
        "PM.pt" => {
            let pm = pm_at(inp);
            let w = inp.v();
            format!(
                "{},{},{},{},{},{},{},{},{}",
                pm.point_on_surface(&w),
                pm.point_on_surface_xyz(w.x, w.y, w.z),
                pm.point_outside(&w),
                pm.point_outside_xyz(w.x, w.y, w.z),
                fv(&pm.create_surface_point(&w)),
                h(pm.minimum_magnitude()),
                h(pm.maximum_magnitude()),
                h(pm.minimum_x_value()),
                h(pm.maximum_z_value())
            )
        }
        "PM.round" => {
            let pm = pm_at(inp);
            let x = inp.f();
            let e = pm.doc_value_encoder();
            [
                e.round_down_x(x),
                e.round_up_x(x),
                e.round_down_y(x),
                e.round_up_y(x),
                e.round_down_z(x),
                e.round_up_z(x),
            ]
            .iter()
            .map(|v| h(*v))
            .collect::<Vec<_>>()
            .join(",")
        }
        "PM.dv" => {
            let pm = pm_at(inp);
            let p = inp.g();
            let e = pm.doc_value_encoder();
            r(e.encode_point(&p).map(|dv| {
                format!(
                    "{dv},{},{},{},{}",
                    h(e.decode_x_value(dv)),
                    h(e.decode_y_value(dv)),
                    h(e.decode_z_value(dv)),
                    fv(&e.decode_point(dv))
                )
            }))
        }
        "B.ops" => {
            let pm = pm_at(inp);
            let flags = inp.i();
            let n = inp.f() as usize;
            let pts: Vec<GeoPoint> = (0..n).map(|_| inp.g()).collect();
            let probe = inp.v();
            let mut b = XYZBounds::new();
            let mut lb = LatLonBounds::new();
            for (i, p) in pts.iter().enumerate() {
                if flags & 1 != 0 && i == 0 {
                    b.add_x_value(p).add_y_value(p).add_z_value(p);
                    lb.add_x_value(p).add_y_value(p).add_z_value(p);
                } else {
                    b.add_point(p);
                    lb.add_point(p);
                }
            }
            if flags & 2 != 0 {
                b.is_wide().no_longitude_bound();
                lb.is_wide();
            }
            if flags & 4 != 0 {
                lb.no_longitude_bound();
            }
            if flags & 8 != 0 {
                lb.no_top_latitude_bound();
                b.no_top_latitude_bound();
            }
            if flags & 16 != 0 {
                lb.no_bottom_latitude_bound();
                b.no_bottom_latitude_bound();
            }
            if flags & 32 != 0 {
                lb.no_bound(&pm);
                b.no_bound(&pm);
            }
            let mut other = XYZBounds::new();
            other.add_point(&GeoPoint::new(probe.x, probe.y, probe.z));
            let mut sum = XYZBounds::new();
            b.add_bounds(&mut sum);
            other.add_bounds(&mut sum);
            format!(
                "{}|{}|{},{},{},{},{},{}{}{}{}{}{}|{}|{}|{}",
                fxb(&b),
                flb(&lb),
                b.is_within(&probe),
                b.is_within_xyz(probe.x, probe.y, probe.z),
                b.overlaps(&other),
                other.overlaps(&b),
                b.overlaps(&sum),
                b.is_smallest_min_x(&pm),
                b.is_largest_max_x(&pm),
                b.is_smallest_min_y(&pm),
                b.is_largest_max_y(&pm),
                b.is_smallest_min_z(&pm),
                b.is_largest_max_z(&pm),
                fxb(&sum),
                b.java_to_string(),
                lb.java_to_string()
            )
        }
        other => panic!("unknown op {other}"),
    }
}

/// `text` with every 16-digit hex double that is a NaN replaced by `NaN`.
///
/// fdlibm's and the planet math's invalid results are `0.0 / 0.0`-style
/// NaNs whose sign is the CPU's default -- negative on x86-64, where the
/// fixture was written, positive on aarch64 -- and Java leaves NaN bits
/// unspecified, so a NaN only has to be a NaN; every other value is still
/// compared bit for bit.
fn nan_blind(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        let run = bytes[i..]
            .iter()
            .take_while(|b| b.is_ascii_hexdigit())
            .count();
        if run == 0 {
            // Hex digits are ASCII, so the next one is a char boundary.
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
fn nan_blind_ignores_only_the_sign_of_a_nan() {
    assert_eq!(
        nan_blind("fff8000000000000,1|7ff8000000000001"),
        "NaN,1|NaN"
    );
    assert_eq!(
        nan_blind("7ff0000000000000 false"),
        "7ff0000000000000 false"
    );
    assert_ne!(nan_blind("3ff0000000000000"), nan_blind("bff0000000000000"));
}

#[test]
fn geo3d_primitives_match_lucene() {
    let text = std::fs::read_to_string(root().join("math.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenGeo3dMath");
    let pms = vec![
        PlanetModel::sphere(),
        PlanetModel::wgs84(),
        PlanetModel::clarke_1866(),
        Arc::new(PlanetModel::new(1.1, 0.9)),
        Arc::new(PlanetModel::new(0.95, 1.05)),
    ];
    let mut failures = Vec::new();
    let mut checked = 0;
    for line in text.lines() {
        let a: Vec<&str> = line.split('\t').collect();
        assert_eq!(a[0], "M");
        let mut inp = In {
            t: a[2].split(' ').collect(),
            at: 0,
        };
        let got = run(a[1], &mut inp, &pms);
        assert_eq!(inp.at, inp.t.len(), "{line}: inputs not all read");
        checked += 1;
        if nan_blind(&got) != nan_blind(a[4]) && failures.len() < 30 {
            failures.push(format!("{line}\n  rust: {got}"));
        }
    }
    assert!(checked > 7000);
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn java_double_text_is_used_for_to_strings() {
    // `Vector.toString` goes through Java's `Double.toString`.
    assert_eq!(
        Vector::new(1.0, 0.5, 1e-7).to_string(),
        format!("[X=1.0, Y=0.5, Z={}]", java_double_string(1e-7))
    );
}
