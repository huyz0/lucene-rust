#![cfg(test)]
use super::*;

#[derive(Default)]
struct Hooks {
    shape_calls: Vec<&'static str>,
    visits: Vec<usize>,
    xs: Vec<f32>,
    ys: Vec<f32>,
    fail_at: Option<usize>,
    invalid_x: bool,
    fail_polygon: bool,
}
impl Hooks {
    fn visit(&mut self, at: usize, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
        assert_eq!(xs.len(), ys.len());
        if self.visits.is_empty() {
            self.xs = xs.to_vec();
            self.ys = ys.to_vec();
        } else {
            assert!(self
                .xs
                .iter()
                .zip(xs)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
            assert!(self
                .ys
                .iter()
                .zip(ys)
                .all(|(a, b)| a.to_bits() == b.to_bits()));
        }
        self.visits.push(at);
        if self.fail_at == Some(at) {
            return Err(GeoError::IllegalArgument(format!("verify {at}")));
        }
        Ok(())
    }
}
impl XYPointCase for Hooks {
    type Error = GeoError;
    fn next_x(&mut self, _: &mut JavaRandom) -> f32 {
        self.shape_calls.push("x");
        if self.invalid_x {
            f32::NAN
        } else {
            9.0
        }
    }
    fn next_y(&mut self, _: &mut JavaRandom) -> f32 {
        self.shape_calls.push("y");
        7.0
    }
    fn next_box(&mut self, _: &mut JavaRandom) -> XYRectangle {
        self.shape_calls.push("box");
        XYRectangle::new(-1.0, 1.0, -2.0, 2.0).unwrap()
    }
    fn next_polygon(&mut self, _: &mut JavaRandom, _: bool) -> Result<XYPolygon, GeoError> {
        self.shape_calls.push("polygon");
        if self.fail_polygon {
            return Err(GeoError::IllegalArgument("polygon override".into()));
        }
        XYPolygon::new(&[0.0, 1.0, 0.0, 0.0], &[0.0, 0.0, 1.0, 0.0], vec![])
    }
    fn verify_random_rectangles(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.visit(0, xs, ys)
    }
    fn verify_random_distances(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.visit(1, xs, ys)
    }
    fn verify_random_polygons(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.visit(2, xs, ys)
    }
    fn verify_random_geometries(
        &mut self,
        _: &mut JavaRandom,
        _: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        self.visit(3, xs, ys)
    }
}

fn generate(
    which: usize,
    hooks: &mut Hooks,
    random: &mut JavaRandom,
    multiplier: i32,
) -> Result<(), GeoError> {
    let mut base = BaseXYPoint::new(hooks, random, multiplier);
    match which {
        0 => base.random_tiny(),
        1 => base.random_medium(),
        2 => base.same_point_many_times(),
        3 => base.low_cardinality(),
        4 => base.all_x_equal(),
        5 => base.all_y_equal(),
        _ => unreachable!(),
    }
}

#[test]
fn verification_errors_stop_at_each_hook_without_new_random_draws() {
    for generator in 0..6 {
        let mut complete = Hooks::default();
        let mut expected_random = JavaRandom::new(42);
        generate(generator, &mut complete, &mut expected_random, 1).unwrap();
        let expected_tail = expected_random.next_int();
        assert_eq!(complete.visits, [0, 1, 2, 3]);
        for fail_at in 0..4 {
            let mut hooks = Hooks {
                fail_at: Some(fail_at),
                ..Hooks::default()
            };
            let mut random = JavaRandom::new(42);
            let error = generate(generator, &mut hooks, &mut random, 1).unwrap_err();
            assert!(
                matches!(error, GeoError::IllegalArgument(message) if message == format!("verify {fail_at}"))
            );
            assert_eq!(hooks.visits, (0..=fail_at).collect::<Vec<_>>());
            assert_eq!(hooks.shape_calls, complete.shape_calls);
            assert_eq!(random.next_int(), expected_tail);
        }
    }
}

#[test]
fn subclass_coordinates_missing_markers_and_all_y_copy_typo_are_retained() {
    for generator in [4, 5] {
        let mut hooks = Hooks::default();
        generate(generator, &mut hooks, &mut JavaRandom::new(42), 1).unwrap();
        assert!(hooks.ys.iter().any(|y| y.is_nan()));
        assert!(hooks
            .xs
            .iter()
            .zip(&hooks.ys)
            .all(|(&x, &y)| !y.is_nan() || x.to_bits() == 0));
        if generator == 4 {
            assert!(hooks
                .xs
                .iter()
                .zip(&hooks.ys)
                .all(|(&x, &y)| y.is_nan() || x == 9.0));
        } else {
            assert!(hooks
                .xs
                .iter()
                .zip(&hooks.ys)
                .any(|(&x, &y)| x.to_bits() == 0 && y == 7.0));
            assert!(hooks
                .xs
                .iter()
                .zip(&hooks.ys)
                .all(|(&x, &y)| y.is_nan() || ((x == 9.0 || x.to_bits() == 0) && y == 7.0)));
        }
    }
    let mut hooks = Hooks::default();
    generate(2, &mut hooks, &mut JavaRandom::new(0), 2).unwrap();
    assert!((2000..=3000).contains(&hooks.xs.len()));
    assert!(hooks.xs.iter().all(|&x| x == 9.0));
    assert!(hooks.ys.iter().all(|&y| y == 7.0));
    assert_eq!(hooks.shape_calls, ["x", "y"]);
    assert_eq!(FIELD_NAME, "point");
}

fn first_geometry_seed(kind: i32) -> i64 {
    (0..1000)
        .find(|&seed| {
            let mut random = JavaRandom::new(seed);
            random.next_int_bounded(4);
            random.next_int_bounded(3) == kind
        })
        .unwrap()
}

#[test]
fn geometry_default_dispatches_overrides_and_propagates_constructor_errors() {
    let mut seen = [false; 3];
    for seed in 0..32 {
        let mut hooks = Hooks::default();
        let mut random = JavaRandom::new(seed);
        let geometries = hooks.next_geometry(&mut random, false).unwrap();
        assert!((1..=4).contains(&geometries.len()));
        let mut expected_calls = Vec::new();
        for geometry in geometries {
            match geometry {
                XYGeometry::Point(p) => {
                    seen[0] = true;
                    assert_eq!((p.x(), p.y()), (9.0, 7.0));
                    expected_calls.extend(["x", "y"]);
                }
                XYGeometry::Rectangle(r) => {
                    seen[1] = true;
                    assert_eq!((r.min_x, r.max_x, r.min_y, r.max_y), (-1.0, 1.0, -2.0, 2.0));
                    expected_calls.push("box");
                }
                XYGeometry::Polygon(p) => {
                    seen[2] = true;
                    assert_eq!(p.poly_x(), [0.0, 1.0, 0.0, 0.0]);
                    expected_calls.push("polygon");
                }
                other => panic!("unexpected geometry {other:?}"),
            }
        }
        assert_eq!(hooks.shape_calls, expected_calls);
    }
    assert_eq!(seen, [true; 3]);
    for kind in [0, 2] {
        let seed = first_geometry_seed(kind);
        let mut hooks = Hooks {
            invalid_x: kind == 0,
            fail_polygon: kind == 2,
            ..Hooks::default()
        };
        let mut random = JavaRandom::new(seed);
        assert!(hooks.next_geometry(&mut random, true).is_err());
        assert_eq!(
            hooks.shape_calls,
            if kind == 0 {
                vec!["x", "y"]
            } else {
                vec!["polygon"]
            }
        );
        let mut expected = JavaRandom::new(seed);
        expected.next_int_bounded(4);
        expected.next_int_bounded(3);
        assert_eq!(
            random.next_int(),
            expected.next_int(),
            "error stops before the next shape"
        );
    }
}

#[test]
fn context_and_oracle_boundaries_remain_explicit() {
    for multiplier in [0, -1, i32::MAX] {
        assert!(std::panic::catch_unwind(|| {
            BaseXYPoint::new(&mut Hooks::default(), &mut JavaRandom::new(0), multiplier);
        })
        .is_err());
    }
    let rect = XYRectangle::new(-2.0, 3.0, -1.0, 4.0).unwrap();
    assert!(rect_contains_point(&rect, -2.0, -1.0));
    assert!(rect_contains_point(&rect, 3.0, 4.0));
    assert!(!rect_contains_point(&rect, 3.0, 4.1));
    assert!(!rect_contains_point(&rect, f64::NAN, 1.0));
    assert!(rect_contains_point(&rect, 1.0, f64::NAN));
    assert_eq!(cartesian_distance(0.0, 0.0, 3.0, 4.0), 5.0);
    assert!(cartesian_distance(f64::MAX, 0.0, -f64::MAX, 0.0).is_infinite());
    assert!(cartesian_distance(f64::INFINITY, 0.0, f64::INFINITY, 0.0).is_nan());
}
