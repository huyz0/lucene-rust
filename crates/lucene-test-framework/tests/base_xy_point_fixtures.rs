//! Actual inherited Java generators with capture-only verification hooks.
use lucene_test_framework::geo::base_xy_point::{
    cartesian_distance, rect_contains_point, BaseXYPoint, XYPointCase,
};
use lucene_util::geo::{GeoError, XYGeometry, XYPolygon, XYRectangle};
use lucene_util::java_random::JavaRandom;

fn fixture(name: &str) -> String {
    std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(format!("base_xy_point_{name}.tsv")),
    )
    .unwrap()
}

#[derive(Default)]
struct Capture {
    xs: Vec<u32>,
    ys: Vec<u32>,
    visits: Vec<&'static str>,
}
impl Capture {
    fn capture(&mut self, name: &'static str, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
        let xbits: Vec<_> = xs.iter().map(|v| v.to_bits()).collect();
        let ybits: Vec<_> = ys.iter().map(|v| v.to_bits()).collect();
        assert_eq!(xs.len(), ys.len());
        if self.visits.is_empty() {
            self.xs = xbits;
            self.ys = ybits;
        } else {
            assert_eq!(self.xs, xbits, "every callback receives the same X array");
            assert_eq!(self.ys, ybits, "every callback receives the same Y array");
        }
        self.visits.push(name);
        Ok(())
    }
}
impl XYPointCase for Capture {
    type Error = GeoError;
    fn verify_random_rectangles(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture("rectangles", xs, ys)
    }
    fn verify_random_distances(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture("distances", xs, ys)
    }
    fn verify_random_polygons(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture("polygons", xs, ys)
    }
    fn verify_random_geometries(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.capture("geometries", xs, ys)
    }
}

fn bits(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| v.to_bits().to_string())
        .collect::<Vec<_>>()
        .join(",")
}
fn parse_bits(text: &str) -> Vec<u32> {
    text.split(',').map(|v| v.parse().unwrap()).collect()
}
fn polygon(value: &XYPolygon) -> String {
    assert_eq!(value.num_holes(), 0);
    format!("polygon:{};{}", bits(value.poly_x()), bits(value.poly_y()))
}
fn rectangle(value: &XYRectangle) -> String {
    format!(
        "rectangle:{}",
        bits(&[value.min_x, value.max_x, value.min_y, value.max_y])
    )
}
fn geometries(values: &[XYGeometry]) -> String {
    values
        .iter()
        .map(|geometry| match geometry {
            XYGeometry::Point(point) => format!("point:{}", bits(&[point.x(), point.y()])),
            XYGeometry::Rectangle(value) => rectangle(value),
            XYGeometry::Polygon(value) => polygon(value),
            other => panic!("nextGeometry must not produce {other:?}"),
        })
        .collect::<Vec<_>>()
        .join("|")
}

#[test]
fn inherited_generators_match_every_coordinate_dispatch_and_rng_tail() {
    for nightly in [false, true] {
        let text = fixture(&format!("generators_{nightly}"));
        let mut lines = text.lines();
        let mut count = 0;
        let mut names = std::collections::BTreeSet::new();
        let mut seeds = std::collections::BTreeSet::new();
        while let Some(header) = lines.next() {
            let fields: Vec<_> = header.split('\t').collect();
            assert_eq!(fields.len(), 6);
            assert_eq!(fields[0], "case");
            let seed: i64 = fields[2].parse().unwrap();
            let mut random = JavaRandom::new(seed);
            let mut capture = Capture::default();
            let mut base = BaseXYPoint::new(&mut capture, &mut random, 1);
            match fields[1] {
                "random-tiny" => base.random_tiny(),
                "random-medium" => base.random_medium(),
                "same-point" => base.same_point_many_times(),
                "low-cardinality" => base.low_cardinality(),
                "all-x" => base.all_x_equal(),
                "all-y" => base.all_y_equal(),
                name => panic!("unknown inherited test {name}"),
            }
            .unwrap();
            assert_eq!(capture.xs.len(), fields[3].parse::<usize>().unwrap());
            assert_eq!(capture.visits.join(","), fields[4]);
            assert_eq!(
                random.next_int(),
                fields[5].parse::<i32>().unwrap(),
                "{header}"
            );
            let xline = lines.next().unwrap().strip_prefix("x\t").unwrap();
            let yline = lines.next().unwrap().strip_prefix("y\t").unwrap();
            assert_eq!(capture.xs, parse_bits(xline), "X: {header}");
            assert_eq!(capture.ys, parse_bits(yline), "Y: {header}");
            names.insert(fields[1]);
            seeds.insert(seed);
            count += 1;
        }
        assert_eq!(count, 120);
        assert_eq!(names.len(), 6);
        assert_eq!(seeds.len(), 20);
        assert!(seeds.contains(&i64::MIN) && seeds.contains(&i64::MAX));
    }
}

#[test]
fn default_shape_hooks_match_both_nightly_modes_and_rng_continuations() {
    for nightly in [false, true] {
        let text = fixture(&format!("hooks_{nightly}"));
        let mut current = None;
        let mut random = JavaRandom::new(0);
        let mut capture = Capture::default();
        let mut count = 0;
        for line in text.lines() {
            let fields: Vec<_> = line.split('\t').collect();
            assert_eq!(fields.len(), 5);
            let seed: i64 = fields[1].parse().unwrap();
            let key = (fields[0], seed);
            if current != Some(key) {
                assert_eq!(fields[2], "0");
                current = Some(key);
                random = JavaRandom::new(seed);
            }
            let actual = match fields[0] {
                "x" => bits(&[capture.next_x(&mut random)]),
                "y" => bits(&[capture.next_y(&mut random)]),
                "box" => rectangle(&capture.next_box(&mut random)),
                "polygon" => polygon(&capture.next_polygon(&mut random, nightly).unwrap()),
                "geometry" => geometries(&capture.next_geometry(&mut random, nightly).unwrap()),
                name => panic!("unknown default hook {name}"),
            };
            assert_eq!(actual, fields[3], "{nightly}: {line}");
            assert_eq!(
                random.next_int(),
                fields[4].parse::<i32>().unwrap(),
                "{line}"
            );
            count += 1;
        }
        assert_eq!(count, 1600);
    }
}

#[test]
fn actual_java_oracles_match_boundaries_nonfinite_points_and_double_arithmetic() {
    let rectangles = [
        XYRectangle::new(-2.0, 3.0, -1.0, 4.0).unwrap(),
        XYRectangle::new(-f32::MAX, f32::MAX, -f32::MAX, f32::MAX).unwrap(),
        XYRectangle::new(-0.0, 0.0, -0.0, 0.0).unwrap(),
    ];
    let value = |text: &str| f64::from_bits(text.parse().unwrap());
    let mut counts = [0, 0];
    for line in fixture("oracles").lines() {
        let fields: Vec<_> = line.split('\t').collect();
        match fields[0] {
            "rectangle" => {
                assert_eq!(fields.len(), 5);
                let rect = &rectangles[fields[1].parse::<usize>().unwrap()];
                assert_eq!(
                    rect_contains_point(rect, value(fields[2]), value(fields[3])),
                    fields[4].parse::<bool>().unwrap(),
                    "{line}"
                );
                counts[0] += 1;
            }
            "distance" => {
                assert_eq!(fields.len(), 6);
                let actual = cartesian_distance(
                    value(fields[1]),
                    value(fields[2]),
                    value(fields[3]),
                    value(fields[4]),
                );
                let expected = value(fields[5]);
                // Arithmetic NaN payloads/signs are not a Java or Rust contract.
                if expected.is_nan() {
                    assert!(actual.is_nan(), "{line}");
                } else {
                    assert_eq!(actual.to_bits(), expected.to_bits(), "{line}");
                }
                counts[1] += 1;
            }
            kind => panic!("unknown oracle {kind}"),
        }
    }
    assert_eq!(counts, [768, 169]);
}
