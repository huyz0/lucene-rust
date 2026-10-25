use lucene_test_framework::geo::shape_test_util::*;
use lucene_util::geo::XYPolygon;
use lucene_util::java_random::JavaRandom;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(format!("shape_test_util_{name}.tsv")),
    )
    .unwrap()
}
fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(",")
}
fn polygon(value: &XYPolygon) -> String {
    format!("{};{}", bits(value.poly_x()), bits(value.poly_y()))
}
#[test]
fn actual_java_seeded_shapes_preserve_every_float_bit_and_rng_draw() {
    for nightly in [false, true] {
        let text = fixture(&nightly.to_string());
        let mut lines = text.lines();
        for seed in 0..128 {
            for op in 0..6 {
                let mut random = JavaRandom::new(seed);
                for at in 0..if op == 0 {
                    32
                } else if op < 4 {
                    8
                } else {
                    4
                } {
                    let value = match op {
                        0 => bits(&[next_float(&mut random)]),
                        1 => {
                            let b = next_box(&mut random);
                            bits(&[b.min_x, b.max_x, b.min_y, b.max_y])
                        }
                        2 => {
                            let p = next_xy_point(&mut random);
                            bits(&[p.x(), p.y()])
                        }
                        3 => {
                            let p = next_circle(&mut random);
                            bits(&[p.x(), p.y(), p.radius()])
                        }
                        4 => polygon(&next_polygon(&mut random, nightly).unwrap()),
                        _ => {
                            let l = next_line(&mut random, nightly).unwrap();
                            format!("{};{}", bits(l.x()), bits(l.y()))
                        }
                    };
                    assert_eq!(
                        format!("{seed}\t{op}\t{at}\t{value}\t{}", random.next_int()),
                        lines.next().unwrap(),
                        "nightly={nightly}"
                    );
                }
            }
        }
        assert!(lines.next().is_none());
    }
}
#[test]
fn actual_java_regular_polygons_preserve_clipping_center_swap_and_invalid_sides() {
    let centers = [
        [1.0, 7.0, 5.0],
        [0.0, 0.0, 0.0],
        [f64::from(f32::MAX), -f64::from(f32::MAX), f64::MAX],
        [1e10, -1e10, 1e8],
        [-0.0, 0.0, 0.25],
        [0.0, 0.0, -0.0],
        [-0.0, -0.0, -0.0],
        [1.0, 7.0, -5.0],
        [f64::NAN, 0.0, 1.0],
        [0.0, f64::NAN, 1.0],
        [0.0, 0.0, f64::NAN],
        [f64::INFINITY, 0.0, 1.0],
        [f64::NEG_INFINITY, 0.0, 1.0],
        [0.0, f64::INFINITY, 1.0],
        [0.0, f64::NEG_INFINITY, 1.0],
        [1.0, 7.0, f64::INFINITY],
        [1.0, 7.0, f64::NEG_INFINITY],
        [f64::from(f32::MAX), -f64::from(f32::MAX), -0.0],
    ];
    for line in fixture("regular").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        let c: usize = fields[0].parse().unwrap();
        let n = fields[1].parse().unwrap();
        let v = centers[c];
        let actual = create_regular_polygon(v[0], v[1], v[2], n)
            .map(|p| polygon(&p))
            .unwrap_or_else(|_| "error".into());
        assert_eq!(actual, fields[2], "center {c}, sides {n}");
    }
}
#[test]
fn actual_java_polygon_oracle_handles_boundaries_nonfinite_points_and_holes_first() {
    let outer = XYPolygon::new(&[0., 4., 4., 0., 0.], &[0., 0., 4., 4., 0.], vec![]).unwrap();
    let triangle = XYPolygon::new(&[-2., 3., 0., -2.], &[0., 0., 5., 0.], vec![]).unwrap();
    let hole = XYPolygon::new(&[1., 2., 2., 1., 1.], &[1., 1., 2., 2., 1.], vec![]).unwrap();
    let holes = XYPolygon::new(outer.poly_x(), outer.poly_y(), vec![hole]).unwrap();
    let polygons = [outer, triangle, holes];
    for line in fixture("contains").lines() {
        let f: Vec<_> = line.split('\t').collect();
        let p: usize = f[0].parse().unwrap();
        let x = f64::from_bits(f[1].parse().unwrap());
        let y = f64::from_bits(f[2].parse().unwrap());
        let actual = contains_slowly(&polygons[p], x, y)
            .map(|b| b.to_string())
            .unwrap_or_else(str::to_owned);
        assert_eq!(actual, f[3], "{line}");
    }
}
