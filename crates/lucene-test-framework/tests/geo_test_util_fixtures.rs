//! `GeoTestUtil`'s polygon, line, nearby-point and nearby-box generators
//! against the actual Java ones: `fixtures/src/testframework/GenGeoTestUtil.java`
//! seeded `random()` with a `java.util.Random` and recorded every result's
//! double bits and the stream's next int.

use lucene_test_framework::geo::geo_test_util::{
    create_regular_polygon, next_box, next_box_near, next_circle_shape, next_line, next_point,
    next_point_near, next_point_near_box, next_polygon, Rect,
};
use lucene_util::geo::{GeoError, Polygon};
use lucene_util::java_random::JavaRandom;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(format!("geo_test_util_{name}.tsv")),
    )
    .expect("run scripts/gen-test-framework-fixtures.sh --only GenGeoTestUtil")
}

fn bits(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn ring(p: &Polygon) -> String {
    format!("{};{}", bits(p.poly_lats()), bits(p.poly_lons()))
}

fn poly(p: &Polygon) -> String {
    let mut s = format!("poly:{}", ring(p));
    for h in p.holes() {
        s.push('|');
        s.push_str(&ring(h));
    }
    s
}

fn rect(r: &Rect) -> String {
    format!(
        "box:{}",
        bits(&[r.min_lat, r.max_lat, r.min_lon, r.max_lon])
    )
}

fn or_error(r: Result<String, GeoError>) -> String {
    r.unwrap_or_else(|e| format!("error:{e}"))
}

/// `GenGeoTestUtil.HOLED`.
fn holed() -> Polygon {
    let hole = Polygon::new(&[-2., 2., 2., -2., -2.], &[-3., -3., 3., 3., -3.], vec![]).unwrap();
    Polygon::new(
        &[-10., -10., 10., 10., -10.],
        &[-20., 20., 20., -20., -20.],
        vec![hole],
    )
    .unwrap()
}

fn result(op: u32, r: &mut JavaRandom) -> String {
    match op {
        0 => or_error(next_polygon(r).map(|p| poly(&p))),
        1 => or_error(next_line(r).map(|l| format!("line:{};{}", bits(l.lats()), bits(l.lons())))),
        2 => or_error(next_polygon(r).map(|p| format!("point:{}", bits(&next_point_near(r, &p))))),
        3 => or_error(next_polygon(r).map(|p| rect(&next_box_near(r, &p)))),
        4 => {
            let b = next_box(r);
            format!("point:{}", bits(&next_point_near_box(r, &b)))
        }
        5 => format!("point:{}", bits(&next_point_near(r, &holed()))),
        6 => rect(&next_box_near(r, &holed())),
        7 => {
            let p = next_point(r);
            let c = next_circle_shape(r);
            format!(
                "circle:{}",
                bits(&[p.lat(), p.lon(), c.lat(), c.lon(), c.radius()])
            )
        }
        _ => unreachable!(),
    }
}

#[test]
fn seeded_generators_match_java_bit_for_bit_with_the_same_draws() {
    let text = fixture("generators");
    let mut rows = 0;
    let mut stream: Option<(u64, u32, JavaRandom)> = None;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (seed, op, at): (u64, u32, u32) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        if at == 0 {
            stream = Some((seed, op, JavaRandom::new(seed as i64)));
        }
        let (s, o, r) = stream.as_mut().unwrap();
        assert_eq!((*s, *o), (seed, op));
        assert_eq!(result(op, r), f[3], "seed {seed} op {op} at {at}");
        assert_eq!(r.next_int(), f[4].parse::<i32>().unwrap(), "{line}");
        rows += 1;
    }
    assert_eq!(rows, 100 * 8 * 4);
}

#[test]
fn regular_polygons_match_java() {
    let centers = [
        (0., 0., 1000.),
        (45., 90., 50_000.),
        (-60., -120., 2_000_000.),
        (89., 0., 500_000.),
        (0., 179.9, 50_000.),
        (10., 10., 0.05),
        (33.3, -44.4, 7_654_321.),
    ];
    let mut rows = 0;
    for line in fixture("regular").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (lat, lon, radius) = centers[f[0].parse::<usize>().unwrap()];
        let gons: usize = f[1].parse().unwrap();
        let got = or_error(create_regular_polygon(lat, lon, radius, gons).map(|p| poly(&p)));
        assert_eq!(got, f[2], "{line}");
        rows += 1;
    }
    assert_eq!(rows, 28);
}
