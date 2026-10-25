//! Bounded `BaseXYPointTestCase` helpers: six inherited point generators,
//! overridable random-shape hooks, four ordered verification callbacks and the
//! rectangle/distance oracles. A NaN Y coordinate marks a missing document.
//!
//! RandomizedContext, TEST_NIGHTLY and RANDOM_MULTIPLIER are explicit arguments.
//! Mutable trait methods replace subclass overrides; owned float vectors replace
//! Java arrays, and `Result` propagates shape/verification exceptions. This seam
//! does not implement the four query verifiers, fixed/multivalued tests, nightly
//! test selection, leaf-codec configuration or query equality/hash assertions.
//! Capturing callbacks consume none of the writer/searcher configuration draws
//! that the full Java verifiers consume. Java's all-Y copy-branch typo is retained.

use lucene_util::geo::{GeoError, XYGeometry, XYPoint, XYPolygon, XYRectangle};
use lucene_util::java_random::JavaRandom;

use super::shape_test_util;
use crate::util::{lucene_test_case, test_util};

/// Java: `BaseXYPointTestCase.FIELD_NAME`.
pub const FIELD_NAME: &str = "point";

/// Subclass hooks used by the bounded generators. Verification receives the
/// same slices in rectangles, distances, polygons, geometries order.
pub trait XYPointCase {
    /// Shape constructor errors and verification failures propagate to the caller.
    type Error: From<GeoError>;

    /// Java: `BaseXYPointTestCase.nextX`.
    fn next_x(&mut self, random: &mut JavaRandom) -> f32 {
        shape_test_util::next_float(random)
    }
    /// Java: `BaseXYPointTestCase.nextY`.
    fn next_y(&mut self, random: &mut JavaRandom) -> f32 {
        shape_test_util::next_float(random)
    }
    /// Java: `BaseXYPointTestCase.nextBox`.
    fn next_box(&mut self, random: &mut JavaRandom) -> XYRectangle {
        shape_test_util::next_box(random)
    }
    /// Java: `BaseXYPointTestCase.nextPolygon`.
    fn next_polygon(
        &mut self,
        random: &mut JavaRandom,
        nightly: bool,
    ) -> Result<XYPolygon, GeoError> {
        shape_test_util::next_polygon(random, nightly)
    }
    /// Java: `BaseXYPointTestCase.nextGeometry`; invokes subclass hooks, choosing
    /// only points, rectangles and polygons (one to four geometries).
    fn next_geometry(
        &mut self,
        random: &mut JavaRandom,
        nightly: bool,
    ) -> Result<Vec<XYGeometry>, GeoError> {
        let len = random.next_int_bounded(4) + 1;
        let mut geometries = Vec::new();
        for _ in 0..len {
            geometries.push(match random.next_int_bounded(3) {
                0 => {
                    let x = self.next_x(random);
                    let y = self.next_y(random);
                    XYGeometry::Point(XYPoint::new(x, y)?)
                }
                1 => XYGeometry::Rectangle(self.next_box(random)),
                _ => XYGeometry::Polygon(self.next_polygon(random, nightly)?),
            });
        }
        Ok(geometries)
    }
    /// Java: `BaseXYPointTestCase.verifyRandomRectangles` override.
    fn verify_random_rectangles(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomDistances` override.
    fn verify_random_distances(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomPolygons` override.
    fn verify_random_polygons(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomGeometries` override.
    fn verify_random_geometries(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), Self::Error>;
}

/// Six inherited generator entry points, parameterized by subclass callbacks.
pub struct BaseXYPoint<'a, C> {
    case: &'a mut C,
    random: &'a mut JavaRandom,
    multiplier: i32,
}

impl<'a, C: XYPointCase> BaseXYPoint<'a, C> {
    /// Java's RANDOM_MULTIPLIER is explicit here. Shape-hook nightly context is
    /// passed directly to `XYPointCase::next_polygon`/`next_geometry`; these six
    /// point-array generators do not use it.
    ///
    /// # Panics
    /// If `multiplier` is not positive, or its scaled array length overflows a
    /// Java int. This bounded API does not model Java negative array allocation.
    pub fn new(case: &'a mut C, random: &'a mut JavaRandom, multiplier: i32) -> Self {
        assert!(multiplier > 0, "multiplier must be positive");
        assert!(
            multiplier <= i32::MAX / 1500,
            "scaled point count exceeds a Java int"
        );
        Self {
            case,
            random,
            multiplier,
        }
    }

    fn at_least(&mut self, count: i32) -> usize {
        usize::try_from(lucene_test_case::at_least(
            self.random,
            self.multiplier,
            count,
        ))
        .expect("positive scaled point count")
    }

    fn next_index(&mut self, bound: usize) -> usize {
        usize::try_from(
            self.random
                .next_int_bounded(i32::try_from(bound).expect("Java array length")),
        )
        .expect("nonnegative array index")
    }

    // Java repeats this rejection loop in all-X, all-Y and doTestRandom.
    fn previous_present(&mut self, ys: &[f32], doc: usize) -> usize {
        loop {
            let old = self.next_index(doc);
            if !ys[old].is_nan() {
                return old;
            }
        }
    }

    // Java: BaseXYPointTestCase.verify. Errors stop dispatch immediately.
    fn verify(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), C::Error> {
        self.case.verify_random_rectangles(xs, ys)?;
        self.case.verify_random_distances(xs, ys)?;
        self.case.verify_random_polygons(xs, ys)?;
        self.case.verify_random_geometries(xs, ys)
    }

    /// Java: `BaseXYPointTestCase.testRandomTiny`.
    pub fn random_tiny(&mut self) -> Result<(), C::Error> {
        self.do_test_random(10)
    }

    /// Java: `BaseXYPointTestCase.testRandomMedium`.
    pub fn random_medium(&mut self) -> Result<(), C::Error> {
        self.do_test_random(1000)
    }

    /// Java: `BaseXYPointTestCase.testSamePointManyTimes`.
    pub fn same_point_many_times(&mut self) -> Result<(), C::Error> {
        let len = self.at_least(1000);
        let x = self.case.next_x(self.random);
        let y = self.case.next_y(self.random);
        self.verify(&vec![x; len], &vec![y; len])
    }

    /// Java: `BaseXYPointTestCase.testLowCardinality`.
    pub fn low_cardinality(&mut self) -> Result<(), C::Error> {
        let len = self.at_least(1000);
        let cardinality = usize::try_from(test_util::next_int(self.random, 2, 20)).expect("2..=20");
        let mut diff_xs = vec![0.0; cardinality];
        let mut diff_ys = vec![0.0; cardinality];
        for i in 0..cardinality {
            diff_xs[i] = self.case.next_x(self.random);
            diff_ys[i] = self.case.next_y(self.random);
        }
        let mut xs = vec![0.0; len];
        let mut ys = vec![0.0; len];
        for i in 0..len {
            let index = self.next_index(cardinality);
            xs[i] = diff_xs[index];
            ys[i] = diff_ys[index];
        }
        self.verify(&xs, &ys)
    }

    /// Java: `BaseXYPointTestCase.testAllYEqual`, including the copy branch
    /// assigning Y from an old X before overwriting Y with the fixed value.
    /// Consequently that branch leaves X at its default positive zero.
    pub fn all_y_equal(&mut self) -> Result<(), C::Error> {
        let len = self.at_least(1000);
        let y = self.case.next_y(self.random);
        let mut xs = vec![0.0; len];
        let mut ys = vec![0.0; len];
        let mut have_real_doc = false;
        for doc in 0..len {
            let x = self.random.next_int_bounded(20);
            if x == 17 {
                ys[doc] = f32::NAN;
                continue;
            }
            if doc > 0 && x == 14 && have_real_doc {
                let old = self.previous_present(&ys, doc);
                ys[doc] = xs[old];
            } else {
                xs[doc] = self.case.next_x(self.random);
                have_real_doc = true;
            }
            ys[doc] = y;
        }
        self.verify(&xs, &ys)
    }

    /// Java: `BaseXYPointTestCase.testAllXEqual`.
    pub fn all_x_equal(&mut self) -> Result<(), C::Error> {
        let len = self.at_least(1000);
        let x = self.case.next_x(self.random);
        let mut xs = vec![0.0; len];
        let mut ys = vec![0.0; len];
        let mut have_real_doc = false;
        for doc in 0..len {
            let pick = self.random.next_int_bounded(20);
            if pick == 17 {
                ys[doc] = f32::NAN;
                continue;
            }
            if doc > 0 && pick == 14 && have_real_doc {
                let old = self.previous_present(&ys, doc);
                ys[doc] = ys[old];
            } else {
                ys[doc] = self.case.next_y(self.random);
                have_real_doc = true;
            }
            xs[doc] = x;
        }
        self.verify(&xs, &ys)
    }

    // Java: BaseXYPointTestCase.doTestRandom. The source's comments swap X/Y:
    // pick 0 retains old Y and draws X; pick 1 retains old X and draws Y.
    fn do_test_random(&mut self, count: i32) -> Result<(), C::Error> {
        let len = self.at_least(count);
        let mut xs = vec![0.0; len];
        let mut ys = vec![0.0; len];
        let mut have_real_doc = false;
        for doc in 0..len {
            let pick = self.random.next_int_bounded(20);
            if pick == 17 {
                ys[doc] = f32::NAN;
                continue;
            }
            if doc > 0 && pick < 3 && have_real_doc {
                let old = self.previous_present(&ys, doc);
                if pick == 0 {
                    ys[doc] = ys[old];
                    xs[doc] = self.case.next_x(self.random);
                } else if pick == 1 {
                    ys[doc] = self.case.next_y(self.random);
                    xs[doc] = xs[old];
                } else {
                    xs[doc] = xs[old];
                    ys[doc] = ys[old];
                }
            } else {
                xs[doc] = self.case.next_x(self.random);
                ys[doc] = self.case.next_y(self.random);
                have_real_doc = true;
            }
        }
        self.verify(&xs, &ys)
    }
}

/// Java: `BaseXYPointTestCase.rectContainsPoint`. No NaN/missing-point guard:
/// NaN Y bypasses the Y comparisons, while NaN X fails the final comparisons.
pub fn rect_contains_point(rect: &XYRectangle, x: f64, y: f64) -> bool {
    if y < f64::from(rect.min_y) || y > f64::from(rect.max_y) {
        return false;
    }
    x >= f64::from(rect.min_x) && x <= f64::from(rect.max_x)
}

/// Java: `BaseXYPointTestCase.cartesianDistance`. Deliberately evaluates double
/// subtraction, two multiplications, addition and sqrt in that order, not hypot.
pub fn cartesian_distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
    let diff_x = x1 - x2;
    let diff_y = y1 - y2;
    (diff_x * diff_x + diff_y * diff_y).sqrt()
}

#[cfg(test)]
mod tests {
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
        fn verify_random_rectangles(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
            self.visit(0, xs, ys)
        }
        fn verify_random_distances(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
            self.visit(1, xs, ys)
        }
        fn verify_random_polygons(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
            self.visit(2, xs, ys)
        }
        fn verify_random_geometries(&mut self, xs: &[f32], ys: &[f32]) -> Result<(), GeoError> {
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
}
