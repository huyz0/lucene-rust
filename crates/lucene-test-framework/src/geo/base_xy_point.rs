//! `BaseXYPointTestCase`: six inherited point generators, overridable
//! random-shape hooks, four ordered verification callbacks and the
//! rectangle/distance oracles; [`XYIndexed`], Java's own verifiers over a
//! subclass's [`XYQueries`], with the multi-valued, inclusive-edge,
//! random-distance, fixed-input (`testSmallSet*`, the box/distance/polygon
//! basics) and invalid-value tests. A NaN Y coordinate marks a missing
//! document.
//!
//! RandomizedContext, TEST_NIGHTLY and RANDOM_MULTIPLIER are explicit
//! arguments. Mutable trait methods replace subclass overrides; owned float
//! vectors replace Java arrays, and `Result` propagates shape exceptions.
//! Every draw the base makes is Java's; as in [`super::base_geo_point`], each
//! index is one segment under a plain `IndexWriter`, skipping the draws of
//! `newIndexWriterConfig`, `RandomIndexWriter` and `newSearcher`, and
//! `testRandomDistance` keeps the default leaf size (drawing Java's).
//! `testEquals` compares the queries a clause wraps by
//! `DocumentQuery::equals`; the `null`-field tests cannot be written
//! against `&str`. Java's all-Y
//! copy-branch typo is retained.

use std::collections::BTreeSet;

use lucene_index::document::{Document, Store, StringField};
use lucene_search::query::Clause;
use lucene_util::geo::{GeoError, XYGeometry, XYPoint, XYPolygon, XYRectangle};
use lucene_util::java_random::JavaRandom;

use super::{point_index, shape_test_util};
use crate::search::test_index::TestIndex;
use crate::util::{lucene_test_case, test_util};

/// Java: `BaseXYPointTestCase.FIELD_NAME`.
pub const FIELD_NAME: &str = "point";

/// Subclass hooks used by the generators. Verification receives the same
/// slices in rectangles, distances, polygons, geometries order, with the
/// test's stream and `RANDOM_MULTIPLIER` (what Java's verifiers draw from);
/// [`XYIndexed`] is Java's own verifiers, over a subclass's [`XYQueries`].
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
    fn verify_random_rectangles(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomDistances` override.
    fn verify_random_distances(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomPolygons` override.
    fn verify_random_polygons(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), Self::Error>;
    /// Java: `BaseXYPointTestCase.verifyRandomGeometries` override.
    fn verify_random_geometries(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), Self::Error>;
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
        self.case
            .verify_random_rectangles(self.random, self.multiplier, xs, ys)?;
        self.case
            .verify_random_distances(self.random, self.multiplier, xs, ys)?;
        self.case
            .verify_random_polygons(self.random, self.multiplier, xs, ys)?;
        self.case
            .verify_random_geometries(self.random, self.multiplier, xs, ys)
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

/// A subclass's field and queries (`BaseXYPointTestCase`'s abstract
/// methods). Each returns Java's `IllegalArgumentException` message as the
/// error.
pub trait XYQueries {
    /// `addPointToDoc(field, doc, x, y)`.
    ///
    /// # Errors
    /// An invalid coordinate.
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        x: f32,
        y: f32,
    ) -> Result<(), String>;
    /// `newRectQuery(field, minX, maxX, minY, maxY)`.
    ///
    /// # Errors
    /// An invalid box.
    fn new_rect_query(
        &self,
        field: &str,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Result<Clause, String>;
    /// `newDistanceQuery(field, centerX, centerY, radius)`.
    ///
    /// # Errors
    /// An invalid centre or radius.
    fn new_distance_query(
        &self,
        field: &str,
        x: f32,
        y: f32,
        radius: f32,
    ) -> Result<Clause, String>;
    /// `newPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// An invalid polygon.
    fn new_polygon_query(&self, field: &str, polygons: &[XYPolygon]) -> Result<Clause, String>;
    /// `newGeometryQuery(field, geometries...)`.
    ///
    /// # Errors
    /// An invalid geometry.
    fn new_geometry_query(&self, field: &str, geometries: &[XYGeometry]) -> Result<Clause, String>;
}

/// A subclass with Java's own verifiers: each indexes the points (a `NaN`
/// coordinate = missing) in one segment, as the geo base does (see
/// [`super::base_geo_point`]), and checks every hit of random boxes,
/// circles, polygons and geometry sets against `rectContainsPoint`,
/// `cartesianDistance`, `ShapeTestUtil.containsSlowly` and the geometries'
/// `Component2D`.
pub struct XYIndexed<Q> {
    /// The subclass's field and queries.
    pub queries: Q,
    /// `TEST_NIGHTLY`, for the shape hooks.
    pub nightly: bool,
}

impl<Q: XYQueries> XYIndexed<Q> {
    fn index_points(&self, random: &mut JavaRandom, xs: &[f32], ys: &[f32]) -> Indexed {
        point_index::index_points(random, xs.len(), |id, doc| {
            if !xs[id].is_nan() && !ys[id].is_nan() {
                self.queries
                    .add_point_to_doc(FIELD_NAME, doc, xs[id], ys[id])
                    .expect("a valid point");
            }
        })
    }

    fn check(
        (index, deleted): &Indexed,
        xs: &[f32],
        ys: &[f32],
        q: Clause,
        what: &str,
        expected: impl Fn(f64, f64) -> bool,
    ) {
        point_index::check(
            index,
            deleted,
            xs.len(),
            q,
            what,
            |id| xs[id].is_nan() || ys[id].is_nan(),
            |id| expected(f64::from(xs[id]), f64::from(ys[id])),
        );
    }
}

type Indexed = (TestIndex, BTreeSet<usize>);

fn at_least(random: &mut JavaRandom, multiplier: i32, i: i32) -> i32 {
    lucene_test_case::at_least(random, multiplier, i)
}

impl<Q: XYQueries> XYPointCase for XYIndexed<Q> {
    type Error = GeoError;

    /// `verifyRandomRectangles`.
    fn verify_random_rectangles(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        let index = self.index_points(random, xs, ys);
        for _ in 0..at_least(random, multiplier, 25) {
            let rect = self.next_box(random);
            let q = self
                .queries
                .new_rect_query(FIELD_NAME, rect.min_x, rect.max_x, rect.min_y, rect.max_y)
                .expect("a valid box");
            Self::check(&index, xs, ys, q, &format!("{rect:?}"), |x, y| {
                rect_contains_point(&rect, x, y)
            });
        }
        Ok(())
    }

    /// `verifyRandomDistances`: circles covering at most half the space.
    fn verify_random_distances(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        let index = self.index_points(random, xs, ys);
        for _ in 0..at_least(random, multiplier, 25) {
            let cx = self.next_x(random);
            let cy = self.next_y(random);
            let mut radius = 0.0;
            while radius == 0.0 {
                radius = random.next_float() * f32::MAX / 2.0;
            }
            let q = self
                .queries
                .new_distance_query(FIELD_NAME, cx, cy, radius)
                .expect("a valid circle");
            let what = format!("centerX={cx} centerY={cy} radius={radius}");
            Self::check(&index, xs, ys, q, &what, |x, y| {
                cartesian_distance(f64::from(cx), f64::from(cy), x, y) <= f64::from(radius)
            });
        }
        Ok(())
    }

    /// `verifyRandomPolygons`: hits against `containsSlowly`.
    fn verify_random_polygons(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        let index = self.index_points(random, xs, ys);
        for _ in 0..at_least(random, multiplier, 75) {
            let polygon = self.next_polygon(random, self.nightly)?;
            let q = self
                .queries
                .new_polygon_query(FIELD_NAME, std::slice::from_ref(&polygon))
                .expect("a valid polygon");
            Self::check(&index, xs, ys, q, &format!("{polygon:?}"), |x, y| {
                shape_test_util::contains_slowly(&polygon, x, y).expect("no holes")
            });
        }
        Ok(())
    }

    /// `verifyRandomGeometries`: hits against the geometries' component.
    fn verify_random_geometries(
        &mut self,
        random: &mut JavaRandom,
        multiplier: i32,
        xs: &[f32],
        ys: &[f32],
    ) -> Result<(), GeoError> {
        let index = self.index_points(random, xs, ys);
        for _ in 0..at_least(random, multiplier, 75) {
            let geometries = self.next_geometry(random, self.nightly)?;
            let q = self
                .queries
                .new_geometry_query(FIELD_NAME, &geometries)
                .expect("valid geometries");
            let component = XYGeometry::create(&geometries)?;
            Self::check(&index, xs, ys, q, &format!("{geometries:?}"), |x, y| {
                component.contains(x, y)
            });
        }
        Ok(())
    }
}

/// `searchSmallSet`'s points (Java's doubles, indexed as floats).
const SMALL_SET: [[f64; 2]; 16] = [
    [32.763_420, -96.774],
    [32.755_952_992_140_7, -96.775_989_532_470_7],
    [32.778_669_420_109_77, -96.777_019_500_732_42],
    [32.775_674_575_542_3, -96.770_603_656_768_8],
    [27.703_618_681_345_585, -139.734_581_708_908_08],
    [32.948_235_888_393_68, -96.453_811_302_781_1],
    [33.060_471_419_708_14, -96.650_848_388_671_88],
    [32.778_650, -96.7772],
    [-88.560_293_717_309_83, -177.235_376_760_363_58],
    [33.541_429_799_076_354, -26.779_373_834_241_003],
    [26.774_024_500_421_728, -77.353_792_761_064_97],
    [-90.0, -14.796_283_808_944_777],
    [32.948_235_888_393_68, -178.853_811_302_781_1],
    [32.948_235_888_393_68, 178.853_811_302_781_1],
    [40.720_611, -73.998_776],
    [-44.5, -179.5],
];

/// The fixed-input and index tests of `BaseXYPointTestCase`, over a
/// subclass with Java's verifiers.
impl<Q: XYQueries> BaseXYPoint<'_, XYIndexed<Q>> {
    fn queries(&self) -> &Q {
        &self.case.queries
    }

    fn add_point(&self, field: &str, doc: &mut Document, x: f32, y: f32) {
        self.queries()
            .add_point_to_doc(field, doc, x, y)
            .expect("a valid point");
    }

    fn rect_query(&self, field: &str, min_x: f32, max_x: f32, min_y: f32, max_y: f32) -> Clause {
        self.queries()
            .new_rect_query(field, min_x, max_x, min_y, max_y)
            .expect("a valid box")
    }

    fn distance_query(&self, field: &str, x: f32, y: f32, radius: f32) -> Clause {
        self.queries()
            .new_distance_query(field, x, y, radius)
            .expect("a valid circle")
    }

    fn polygon_query(&self, field: &str, polygons: &[XYPolygon]) -> Clause {
        self.queries()
            .new_polygon_query(field, polygons)
            .expect("a valid polygon")
    }

    /// `testEquals`: the box queries equal their twins, and the distance
    /// and polygon queries also differ from the same query on another
    /// field.
    pub fn equals(&mut self) {
        let rect = self.case.next_box(self.random);
        let (min_x, max_x, min_y, max_y) = (rect.min_x, rect.max_x, rect.min_y, rect.max_y);
        let q1 = self.rect_query("field", min_x, max_x, min_y, max_y);
        let q2 = self.rect_query("field", min_x, max_x, min_y, max_y);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");

        let x = self.case.next_x(self.random);
        let y = self.case.next_y(self.random);
        let q1 = self.distance_query("field", x, y, 10000.0);
        let q2 = self.distance_query("field", x, y, 10000.0);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");
        let other = self.distance_query("field2", x, y, 10000.0);
        assert!(
            !point_index::query_equals(&q1, &other),
            "{q1:?} == {other:?}"
        );

        let xs = [min_x, max_x, max_x, min_x, min_x];
        let ys = [min_y, min_y, max_y, max_y, min_y];
        let polygon = || XYPolygon::new(&xs, &ys, Vec::new()).expect("a box's polygon");
        let q1 = self.polygon_query("field", &[polygon()]);
        let q2 = self.polygon_query("field", &[polygon()]);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");
        let other = self.polygon_query("field2", &[polygon()]);
        assert!(
            !point_index::query_equals(&q1, &other),
            "{q1:?} == {other:?}"
        );
    }

    /// `testRandomBig` (nightly).
    ///
    /// # Errors
    /// A shape hook's.
    pub fn random_big(&mut self) -> Result<(), GeoError> {
        self.do_test_random(200_000)
    }

    /// `testMultiValued`: two points a document, hits for either.
    pub fn multi_valued(&mut self) {
        let n = self.at_least(1000);
        let (mut xs, mut ys) = (vec![0.0; 2 * n], vec![0.0; 2 * n]);
        let mut docs = Vec::with_capacity(n);
        for id in 0..n {
            let mut doc = Document::new();
            doc.add(StringField::new("id", id.to_string(), Store::Yes));
            for k in [2 * id, 2 * id + 1] {
                xs[k] = self.case.next_x(self.random);
                ys[k] = self.case.next_y(self.random);
                self.add_point(FIELD_NAME, &mut doc, xs[k], ys[k]);
            }
            docs.push(doc);
        }
        let _force_merge = self.random.next_boolean();
        let index = point_index::build(docs, &[]);
        let iters = self.at_least(25);
        for _ in 0..iters {
            let rect = self.case.next_box(self.random);
            let q = self.rect_query(FIELD_NAME, rect.min_x, rect.max_x, rect.min_y, rect.max_y);
            let either = |id: usize| {
                [2 * id, 2 * id + 1]
                    .iter()
                    .any(|&k| rect_contains_point(&rect, f64::from(xs[k]), f64::from(ys[k])))
            };
            let what = format!("{rect:?}");
            point_index::check(&index, &BTreeSet::new(), n, q, &what, |_| false, either);
        }
    }

    /// `testRectBoundariesAreInclusive`: points on a box's corners and edge
    /// midpoints match it, and still match it one ulp larger.
    pub fn rect_boundaries_are_inclusive(&mut self) {
        let rect = shape_test_util::next_box(self.random);
        let mid = |a: f32, b: f32| {
            #[allow(clippy::cast_possible_truncation)]
            let m = ((f64::from(a) + f64::from(b)) / 2.0) as f32;
            m
        };
        let mut docs = Vec::new();
        for i in 0..3 {
            let y = match i {
                0 => rect.min_y,
                1 => mid(rect.min_y, rect.max_y),
                _ => rect.max_y,
            };
            for j in 0..3 {
                let x = match j {
                    0 => rect.min_x,
                    1 if i == 1 => continue,
                    1 => mid(rect.min_x, rect.max_x),
                    _ => rect.max_x,
                };
                let mut doc = Document::new();
                self.add_point(FIELD_NAME, &mut doc, x, y);
                docs.push(doc);
            }
        }
        let index = point_index::build(docs, &[]);
        let count = |b: [f32; 4]| {
            point_index::hits(&index, self.rect_query(FIELD_NAME, b[0], b[1], b[2], b[3])).len()
        };
        let exact = [rect.min_x, rect.max_x, rect.min_y, rect.max_y];
        assert_eq!(count(exact), 8, "exact edges of {rect:?}");
        #[allow(clippy::float_cmp)]
        for (k, limit) in [(0, -f32::MAX), (1, f32::MAX), (2, -f32::MAX), (3, f32::MAX)] {
            if exact[k] != limit {
                let mut b = exact;
                b[k] = if k % 2 == 0 {
                    b[k].next_down()
                } else {
                    b[k].next_up()
                };
                assert_eq!(count(b), 8, "{b:?}");
            }
        }
    }

    /// `testRandomDistance`: a few ten-document indexes, a hundred circles
    /// each.
    pub fn random_distance(&mut self) {
        let iters = self.at_least(1);
        for _ in 0..iters {
            self.do_random_distance_test(10, 100);
        }
    }

    /// `testRandomDistanceHuge` (nightly).
    pub fn random_distance_huge(&mut self) {
        for _ in 0..10 {
            self.do_random_distance_test(2000, 100);
        }
    }

    /// `doRandomDistanceTest(numDocs, numQueries)`: hits against the
    /// cartesian distance; the default points format where Java installs
    /// 2-5 points per leaf (the size is still drawn).
    pub fn do_random_distance_test(&mut self, num_docs: usize, num_queries: usize) {
        let _points_in_leaf = 2 + self.random.next_int_bounded(4);
        let mut points = Vec::with_capacity(num_docs);
        let mut docs = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let x = self.case.next_x(self.random);
            let y = self.case.next_y(self.random);
            let mut doc = Document::new();
            self.add_point("field", &mut doc, x, y);
            docs.push(doc);
            points.push((f64::from(x), f64::from(y)));
        }
        let index = point_index::build(docs, &[]);
        for _ in 0..num_queries {
            let circle = shape_test_util::next_circle(self.random);
            let (x, y, radius) = (circle.x(), circle.y(), circle.radius());
            let expected: BTreeSet<usize> = (0..num_docs)
                .filter(|&d| {
                    cartesian_distance(f64::from(x), f64::from(y), points[d].0, points[d].1)
                        <= f64::from(radius)
                })
                .collect();
            let actual = point_index::hits(&index, self.distance_query("field", x, y, radius));
            assert_eq!(actual, expected, "center: ({x},{y}), radius={radius}");
        }
    }

    /// `testBoxBasics`, `testDistanceBasics`, `testPolygonBasics`,
    /// `testPolygonHole`, `testPolygonHoleExcludes` and
    /// `testMultiPolygonBasics`: one point, found (or excluded by a hole).
    // Java's float literals, as written.
    #[allow(clippy::excessive_precision)]
    pub fn basics(&mut self) {
        let square = |xs: [f32; 2], ys: [f32; 2], holes: Vec<XYPolygon>| {
            XYPolygon::new(
                &[xs[0], xs[0], xs[1], xs[1], xs[0]],
                &[ys[0], ys[1], ys[1], ys[0], ys[0]],
                holes,
            )
            .expect("a valid polygon")
        };
        let outer = |hole: Option<XYPolygon>| {
            square([18.0, 19.0], [-66.0, -65.0], hole.into_iter().collect())
        };
        let cases = [
            ("box", self.rect_query("field", 18.0, 19.0, -66.0, -65.0), 1),
            (
                "distance",
                self.distance_query("field", 18.0, -65.0, 20.0),
                1,
            ),
            ("polygon", self.polygon_query("field", &[outer(None)]), 1),
            (
                "hole",
                self.polygon_query(
                    "field",
                    &[outer(Some(square([18.5, 18.7], [-65.7, -65.4], vec![])))],
                ),
                1,
            ),
            (
                "excluding hole",
                self.polygon_query(
                    "field",
                    &[outer(Some(square([18.2, 18.4], [-65.3, -65.2], vec![])))],
                ),
                0,
            ),
            (
                "multi-polygon",
                self.polygon_query(
                    "field",
                    &[square([28.0, 29.0], [-56.0, -55.0], vec![]), outer(None)],
                ),
                1,
            ),
        ];
        for (what, q, want) in cases {
            let mut doc = Document::new();
            self.add_point("field", &mut doc, 18.313_694, -65.227_444);
            assert_eq!(point_index::count(vec![doc], q), want, "{what}");
        }
    }

    /// `testIndexExtremeValues`, `testIndexNaNValues` and
    /// `testIndexInfValues`.
    pub fn index_values(&mut self) {
        let mut doc = Document::new();
        let m = f32::MAX;
        for (x, y) in [(m, m), (m, -m), (-m, m), (-m, -m)] {
            self.add_point("foo", &mut doc, x, y);
        }
        for (x, y) in [
            (f32::NAN, 50.0),
            (50.0, f32::NAN),
            (f32::INFINITY, 0.0),
            (f32::NEG_INFINITY, 0.0),
            (0.0, f32::INFINITY),
            (0.0, f32::NEG_INFINITY),
        ] {
            let result = self.queries().add_point_to_doc("foo", &mut doc, x, y);
            point_index::assert_rejected(result, &["invalid value"]);
        }
    }

    /// `testBoxInvalidCoordinates`, `testDistanceIllegal`,
    /// `testDistanceNegative`, `testDistanceNaN` and `testDistanceInf`.
    pub fn invalid_queries(&mut self) {
        let nan = f32::NAN;
        let q = self.queries();
        point_index::assert_rejected(q.new_rect_query("field", nan, nan, nan, nan), &[]);
        point_index::assert_rejected(q.new_distance_query("field", nan, nan, 120_000.0), &[]);
        for (radius, parts) in [
            (-1.0, &["radius"][..]),
            (nan, &["radius", "NaN"][..]),
            (f32::INFINITY, &["radius", "finite"][..]),
            (f32::NEG_INFINITY, &["radius", "bigger than 0"][..]),
        ] {
            point_index::assert_rejected(q.new_distance_query("field", 18.0, 19.0, radius), parts);
        }
    }

    /// `searchSmallSet(query, size)`'s hit count: sixteen points in
    /// `"point"`, eight documents holding two of them, and up to nine
    /// documents without.
    fn search_small_set(&mut self, query: Clause) -> usize {
        #[allow(clippy::cast_possible_truncation)]
        let point = |p: [f64; 2]| (p[0] as f32, p[1] as f32);
        let mut docs = Vec::new();
        for p in SMALL_SET {
            let mut doc = Document::new();
            let (x, y) = point(p);
            self.add_point("point", &mut doc, x, y);
            docs.push(doc);
        }
        for pair in SMALL_SET.chunks(2) {
            let mut doc = Document::new();
            for &p in pair {
                let (x, y) = point(p);
                self.add_point("point", &mut doc, x, y);
            }
            docs.push(doc);
        }
        point_index::small_set_string_docs(self.random, &mut docs);
        point_index::count(docs, query)
    }

    /// The `testSmallSet*` family over [`Self::search_small_set`].
    // Java's float literals, as written.
    #[allow(clippy::excessive_precision)]
    pub fn small_set(&mut self) {
        let m = f32::MAX;
        let poly = XYPolygon::new(
            &[
                33.073130, 32.9942669, 32.938386, 33.0374494, 33.1369762, 33.1162747, 33.073130,
                33.073130,
            ],
            &[
                -96.7682647,
                -96.8280029,
                -96.6288757,
                -96.4929199,
                -96.6041564,
                -96.7449188,
                -96.76826477,
                -96.7682647,
            ],
            vec![],
        )
        .expect("a valid polygon");
        let whole = XYPolygon::new(&[-m, m, m, -m, -m], &[-m, -m, m, m, -m], vec![])
            .expect("a valid polygon");
        let (cx, cy) = (32.94823588839368_f32, -96.4538113027811_f32);
        let cases: Vec<(&str, Clause, usize)> = vec![
            (
                "Rect",
                self.rect_query("point", 32.778, 32.779, -96.778, -96.777),
                4,
            ),
            (
                "Rect2",
                self.rect_query("point", -45.0, -44.0, -180.0, 180.0),
                2,
            ),
            // 3 single-valued docs + 2 multi-valued docs.
            (
                "MultiValued",
                self.rect_query("point", 32.755, 32.776, -180.0, 180.77),
                5,
            ),
            ("WholeSpace", self.rect_query("point", -m, m, -m, m), 24),
            ("Poly", self.polygon_query("point", &[poly]), 2),
            ("PolyWholeSpace", self.polygon_query("point", &[whole]), 24),
            ("Distance", self.distance_query("point", cx, cy, 6.0), 11),
            (
                "TinyDistance",
                self.distance_query("point", 40.720611, -73.998776, 0.1),
                2,
            ),
            ("HugeDistance", self.distance_query("point", cx, cy, m), 24),
        ];
        for (what, q, want) in cases {
            assert_eq!(self.search_small_set(q), want, "testSmallSet{what}");
        }
    }
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
}
