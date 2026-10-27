//! `BaseGeoPointTestCase`: random points -- some missing, some sharing a
//! latitude, a longitude or the whole point with an earlier document --
//! indexed by a subclass's field, then queried by random boxes, circles,
//! polygons and geometry sets, every hit checked against
//! `rectContainsPoint`, `haversinMeters`, `GeoTestUtil.containsSlowly` or
//! the geometries' `Component2D`; multi-valued documents; inclusive box
//! edges; the fixed-input tests (`testBoxBasics`, the `testSmallSet*`
//! family, the polygon and hole basics, the north-pole sliver); and the
//! invalid-value checks.
//!
//! Every draw the base makes is Java's, in Java's order. What it hands
//! to Lucene's test infrastructure is not: as in
//! [`crate::search::base_range_field`], each index is one segment under a
//! plain `IndexWriter` with the deletes in one last commit, so the draws
//! `newIndexWriterConfig`, `RandomIndexWriter` and `newSearcher` make are
//! skipped and the `forceMerge` coin is drawn and ignored.
//! `testRandomDistance` keeps the default points format where Java
//! installs one with 2-5 points per leaf (it still draws that size).
//! `testEquals` compares the queries a clause wraps by
//! `DocumentQuery::equals`. The `null`-field tests cannot be written
//! against `&str` (see the ledger).

use std::collections::BTreeSet;

use lucene_index::document::{Document, Store, StringField};
use lucene_search::query::Clause;
use lucene_util::geo::{GeoEncodingUtils, GeoUtils, LatLonGeometry, Polygon, Rectangle};
use lucene_util::java_random::JavaRandom;
use lucene_util::sloppy_math::haversin_meters;

use super::geo_test_util::{self, Rect};
use super::point_index;
use crate::search::test_index::TestIndex;
use crate::util::{lucene_test_case, test_util};

/// `BaseGeoPointTestCase.FIELD_NAME`.
pub const FIELD_NAME: &str = "point";

/// A subclass: the field it indexes and the queries it answers with. Each
/// returns Java's `IllegalArgumentException` message as the error.
pub trait GeoPointCase {
    /// `addPointToDoc(field, doc, lat, lon)`.
    ///
    /// # Errors
    /// An invalid coordinate.
    fn add_point_to_doc(
        &self,
        field: &str,
        doc: &mut Document,
        lat: f64,
        lon: f64,
    ) -> Result<(), String>;
    /// `newRectQuery(field, minLat, maxLat, minLon, maxLon)`.
    ///
    /// # Errors
    /// An invalid coordinate.
    fn new_rect_query(&self, field: &str, rect: &Rect) -> Result<Clause, String>;
    /// `newDistanceQuery(field, centerLat, centerLon, radiusMeters)`.
    ///
    /// # Errors
    /// An invalid centre or radius.
    fn new_distance_query(
        &self,
        field: &str,
        lat: f64,
        lon: f64,
        radius_meters: f64,
    ) -> Result<Clause, String>;
    /// `newPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// An invalid polygon.
    fn new_polygon_query(&self, field: &str, polygons: &[Polygon]) -> Result<Clause, String>;
    /// `newGeometryQuery(field, geometries...)`.
    ///
    /// # Errors
    /// An invalid geometry.
    fn new_geometry_query(
        &self,
        field: &str,
        geometries: &[LatLonGeometry],
    ) -> Result<Clause, String>;
    /// `quantizeLat(lat)`: identity unless the index rounds.
    fn quantize_lat(&self, lat: f64) -> f64 {
        lat
    }
    /// `quantizeLon(lon)`.
    fn quantize_lon(&self, lon: f64) -> f64 {
        lon
    }
}

/// `rectContainsPoint(rect, lat, lon)`: inclusive, and across the dateline
/// when `max_lon < min_lon`.
///
/// # Panics
/// On a `NaN` latitude.
#[must_use]
pub fn rect_contains_point(rect: &Rect, lat: f64, lon: f64) -> bool {
    assert!(!lat.is_nan(), "a missing point");
    if lat < rect.min_lat || lat > rect.max_lat {
        return false;
    }
    if rect.min_lon <= rect.max_lon {
        lon >= rect.min_lon && lon <= rect.max_lon
    } else {
        lon <= rect.max_lon || lon >= rect.min_lon
    }
}

/// One subclass's test methods over one stream (`NaN` latitude = missing).
pub struct BaseGeoPoint<'a, C> {
    case: C,
    random: &'a mut JavaRandom,
    multiplier: i32,
}

impl<'a, C: GeoPointCase> BaseGeoPoint<'a, C> {
    /// `case` drawing from `random`, under `multiplier`.
    pub fn new(case: C, random: &'a mut JavaRandom, multiplier: i32) -> Self {
        Self {
            case,
            random,
            multiplier,
        }
    }

    fn at_least(&mut self, i: i32) -> usize {
        usize::try_from(lucene_test_case::at_least(self.random, self.multiplier, i))
            .expect("non-negative")
    }

    fn next_index(&mut self, bound: usize) -> usize {
        let b = i32::try_from(bound).expect("an int bound");
        usize::try_from(self.random.next_int_bounded(b)).expect("non-negative")
    }

    /// `testRandomTiny`: one leaf.
    pub fn random_tiny(&mut self) {
        self.do_test_random(10);
    }

    /// `testRandomMedium`.
    pub fn random_medium(&mut self) {
        self.do_test_random(1000);
    }

    /// `testSamePointManyTimes`.
    pub fn same_point_many_times(&mut self) {
        let (lats, lons) = self.same_point_points();
        self.verify(lats, lons);
    }

    /// `testSamePointManyTimes`'s points, before `verify`.
    pub fn same_point_points(&mut self) -> (Vec<f64>, Vec<f64>) {
        let n = self.at_least(1000);
        let lat = geo_test_util::next_latitude(self.random);
        let lon = geo_test_util::next_longitude(self.random);
        (vec![lat; n], vec![lon; n])
    }

    /// `testLowCardinality`: a tricky adversary for the BKD tree.
    pub fn low_cardinality(&mut self) {
        let (lats, lons) = self.low_cardinality_points();
        self.verify(lats, lons);
    }

    /// `testLowCardinality`'s points, before `verify`.
    pub fn low_cardinality_points(&mut self) -> (Vec<f64>, Vec<f64>) {
        let n = self.at_least(1000);
        let cardinality = usize::try_from(test_util::next_int(self.random, 2, 20)).expect("2..=20");
        let mut diff = Vec::with_capacity(cardinality);
        for _ in 0..cardinality {
            let lat = geo_test_util::next_latitude(self.random);
            let lon = geo_test_util::next_longitude(self.random);
            diff.push((lat, lon));
        }
        let (mut lats, mut lons) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let (lat, lon) = diff[self.next_index(cardinality)];
            lats.push(lat);
            lons.push(lon);
        }
        (lats, lons)
    }

    /// `testAllLatEqual`.
    pub fn all_lat_equal(&mut self) {
        let (lats, lons) = self.all_one_equal_points(true);
        self.verify(lats, lons);
    }

    /// `testAllLonEqual`.
    pub fn all_lon_equal(&mut self) {
        let (lats, lons) = self.all_one_equal_points(false);
        self.verify(lats, lons);
    }

    /// `testAllLatEqual`'s (`lat_fixed`) or `testAllLonEqual`'s points,
    /// before `verify`: some docs missing, some copying an earlier doc's
    /// free coordinate.
    pub fn all_one_equal_points(&mut self, lat_fixed: bool) -> (Vec<f64>, Vec<f64>) {
        let n = self.at_least(1000);
        let fixed = if lat_fixed {
            geo_test_util::next_latitude(self.random)
        } else {
            geo_test_util::next_longitude(self.random)
        };
        let (mut lats, mut lons) = (vec![0.0; n], vec![0.0; n]);
        let mut have_real_doc = false;
        for doc in 0..n {
            let x = self.random.next_int_bounded(20);
            if x == 17 {
                lats[doc] = f64::NAN;
                continue;
            }
            let free = if doc > 0 && x == 14 && have_real_doc {
                let old = loop {
                    let old = self.next_index(doc);
                    if !lats[old].is_nan() {
                        break old;
                    }
                };
                if lat_fixed {
                    lons[old]
                } else {
                    lats[old]
                }
            } else {
                have_real_doc = true;
                if lat_fixed {
                    geo_test_util::next_longitude(self.random)
                } else {
                    geo_test_util::next_latitude(self.random)
                }
            };
            if lat_fixed {
                (lats[doc], lons[doc]) = (fixed, free);
            } else {
                (lats[doc], lons[doc]) = (free, fixed);
            }
        }
        (lats, lons)
    }

    /// `doTestRandom(count)`.
    pub fn do_test_random(&mut self, count: i32) {
        let (lats, lons) = self.random_points(count);
        self.verify(lats, lons);
    }

    /// `doTestRandom(count)`'s points, before `verify`.
    pub fn random_points(&mut self, count: i32) -> (Vec<f64>, Vec<f64>) {
        let n = self.at_least(count);
        let (mut lats, mut lons) = (vec![0.0; n], vec![0.0; n]);
        let mut have_real_doc = false;
        for id in 0..n {
            let x = self.random.next_int_bounded(20);
            if x == 17 {
                lats[id] = f64::NAN;
                continue;
            }
            if id > 0 && x < 3 && have_real_doc {
                let old = loop {
                    let old = self.next_index(id);
                    if !lats[old].is_nan() {
                        break old;
                    }
                };
                match x {
                    0 => {
                        lats[id] = lats[old];
                        lons[id] = geo_test_util::next_longitude(self.random);
                    }
                    1 => {
                        lats[id] = geo_test_util::next_latitude(self.random);
                        lons[id] = lons[old];
                    }
                    _ => {
                        lats[id] = lats[old];
                        lons[id] = lons[old];
                    }
                }
            } else {
                lats[id] = geo_test_util::next_latitude(self.random);
                lons[id] = geo_test_util::next_longitude(self.random);
                have_real_doc = true;
            }
        }
        (lats, lons)
    }

    /// `testRandomBig` (nightly).
    pub fn random_big(&mut self) {
        self.do_test_random(200_000);
    }

    /// `testMultiValued`: two points a document, hits for either.
    pub fn multi_valued(&mut self) {
        let n = self.at_least(1000);
        let (mut lats, mut lons) = (vec![0.0; 2 * n], vec![0.0; 2 * n]);
        let mut docs = Vec::with_capacity(n);
        for id in 0..n {
            let mut doc = Document::new();
            doc.add(StringField::new("id", id.to_string(), Store::Yes));
            for k in [2 * id, 2 * id + 1] {
                lats[k] = self
                    .case
                    .quantize_lat(geo_test_util::next_latitude(self.random));
                lons[k] = self
                    .case
                    .quantize_lon(geo_test_util::next_longitude(self.random));
                self.add_point(FIELD_NAME, &mut doc, lats[k], lons[k]);
            }
            docs.push(doc);
        }
        let _force_merge = self.random.next_boolean();
        let index = point_index::build(docs, &[]);
        let iters = self.at_least(25);
        for _ in 0..iters {
            let rect = geo_test_util::next_box(self.random);
            let q = self.rect_query(FIELD_NAME, &rect);
            Self::check(
                &index,
                &BTreeSet::new(),
                &lats[..n],
                q,
                &format!("{rect:?}"),
                |id| {
                    rect_contains_point(&rect, lats[2 * id], lons[2 * id])
                        || rect_contains_point(&rect, lats[2 * id + 1], lons[2 * id + 1])
                },
            );
        }
    }

    /// `verify(lats, lons)`: quantize as the index does, then the
    /// rectangle, distance, polygon and geometry verifies.
    pub fn verify(&mut self, mut lats: Vec<f64>, mut lons: Vec<f64>) {
        for (lat, lon) in lats.iter_mut().zip(lons.iter_mut()) {
            if !lat.is_nan() {
                *lat = self.case.quantize_lat(*lat);
                *lon = self.case.quantize_lon(*lon);
            }
        }
        self.verify_random_rectangles(&lats, &lons);
        self.verify_random_distances(&lats, &lons);
        self.verify_random_polygons(&lats, &lons);
        self.verify_random_geometries(&lats, &lons);
    }

    fn add_point(&self, field: &str, doc: &mut Document, lat: f64, lon: f64) {
        self.case
            .add_point_to_doc(field, doc, lat, lon)
            .expect("a valid point");
    }

    fn rect_query(&self, field: &str, rect: &Rect) -> Clause {
        self.case.new_rect_query(field, rect).expect("a valid box")
    }

    fn distance_query(&self, field: &str, lat: f64, lon: f64, radius: f64) -> Clause {
        self.case
            .new_distance_query(field, lat, lon, radius)
            .expect("a valid circle")
    }

    fn polygon_query(&self, field: &str, polygons: &[Polygon]) -> Clause {
        self.case
            .new_polygon_query(field, polygons)
            .expect("a valid polygon")
    }

    /// `indexPoints`.
    fn index_points(&mut self, lats: &[f64], lons: &[f64]) -> (TestIndex, BTreeSet<usize>) {
        let case = &self.case;
        point_index::index_points(self.random, lats.len(), |id, doc| {
            if !lats[id].is_nan() {
                case.add_point_to_doc(FIELD_NAME, doc, lats[id], lons[id])
                    .expect("a valid point");
            }
        })
    }

    /// [`point_index::check`] with a `NaN` latitude as missing.
    fn check(
        index: &TestIndex,
        deleted: &BTreeSet<usize>,
        lats: &[f64],
        q: Clause,
        what: &str,
        expected: impl Fn(usize) -> bool,
    ) {
        point_index::check(
            index,
            deleted,
            lats.len(),
            q,
            what,
            |id| lats[id].is_nan(),
            expected,
        );
    }

    /// `verifyRandomRectangles`.
    fn verify_random_rectangles(&mut self, lats: &[f64], lons: &[f64]) {
        let (index, deleted) = self.index_points(lats, lons);
        let iters = self.at_least(25);
        for _ in 0..iters {
            let rect = geo_test_util::next_box(self.random);
            let q = self.rect_query(FIELD_NAME, &rect);
            Self::check(&index, &deleted, lats, q, &format!("{rect:?}"), |id| {
                rect_contains_point(&rect, lats[id], lons[id])
            });
        }
    }

    /// `verifyRandomDistances`: circles covering at most half the earth.
    fn verify_random_distances(&mut self, lats: &[f64], lons: &[f64]) {
        let (index, deleted) = self.index_points(lats, lons);
        let iters = self.at_least(25);
        for _ in 0..iters {
            let lat = geo_test_util::next_latitude(self.random);
            let lon = geo_test_util::next_longitude(self.random);
            let radius = geo_test_util::next_radius_meters(self.random);
            debug_assert!(radius <= GeoUtils::EARTH_MEAN_RADIUS_METERS * std::f64::consts::PI);
            let q = self.distance_query(FIELD_NAME, lat, lon, radius);
            let what = format!("centerLat={lat} centerLon={lon} radiusMeters={radius}");
            Self::check(&index, &deleted, lats, q, &what, |id| {
                haversin_meters(lat, lon, lats[id], lons[id]) <= radius
            });
        }
    }

    /// `nextPolygon()`.
    pub fn next_polygon(&mut self) -> Polygon {
        geo_test_util::next_polygon(self.random).expect("GeoTestUtil makes valid polygons")
    }

    /// `nextGeometry()`: one to four boxes, circles and polygons.
    pub fn next_geometry(&mut self) -> Vec<LatLonGeometry> {
        let length = self.random.next_int_bounded(4) + 1;
        (0..length)
            .map(|_| match self.random.next_int_bounded(3) {
                0 => {
                    let r = geo_test_util::next_box(self.random);
                    LatLonGeometry::Rectangle(
                        Rectangle::new(r.min_lat, r.max_lat, r.min_lon, r.max_lon)
                            .expect("a valid box"),
                    )
                }
                1 => LatLonGeometry::Circle(geo_test_util::next_circle_shape(self.random)),
                _ => LatLonGeometry::Polygon(self.next_polygon()),
            })
            .collect()
    }

    /// `verifyRandomPolygons`: hits against `containsSlowly`.
    fn verify_random_polygons(&mut self, lats: &[f64], lons: &[f64]) {
        let (index, deleted) = self.index_points(lats, lons);
        let iters = self.at_least(75);
        for _ in 0..iters {
            let polygon = self.next_polygon();
            let q = self.polygon_query(FIELD_NAME, std::slice::from_ref(&polygon));
            Self::check(&index, &deleted, lats, q, &format!("{polygon:?}"), |id| {
                geo_test_util::contains_slowly(&polygon, lats[id], lons[id]).expect("no holes")
            });
        }
    }

    /// `verifyRandomGeometries`: hits against the geometries' component.
    fn verify_random_geometries(&mut self, lats: &[f64], lons: &[f64]) {
        let (index, deleted) = self.index_points(lats, lons);
        let iters = self.at_least(75);
        for _ in 0..iters {
            let geometries = self.next_geometry();
            let q = self
                .case
                .new_geometry_query(FIELD_NAME, &geometries)
                .expect("valid geometries");
            let component = LatLonGeometry::create(&geometries).expect("valid geometries");
            Self::check(
                &index,
                &deleted,
                lats,
                q,
                &format!("{geometries:?}"),
                |id| {
                    component.contains(
                        self.case.quantize_lon(lons[id]),
                        self.case.quantize_lat(lats[id]),
                    )
                },
            );
        }
    }

    /// `testRectBoundariesAreInclusive`: points on a quantized box's
    /// corners and edge midpoints match it, still match one ulp larger,
    /// and none match one ulp smaller.
    pub fn rect_boundaries_are_inclusive(&mut self) {
        let r = loop {
            let r = geo_test_util::next_box(self.random);
            if r.max_lon >= r.min_lon {
                break r;
            }
        };
        let rect = Rect {
            min_lat: self.case.quantize_lat(r.min_lat),
            max_lat: self.case.quantize_lat(r.max_lat),
            min_lon: self.case.quantize_lon(r.min_lon),
            max_lon: self.case.quantize_lon(r.max_lon),
        };
        let mut docs = Vec::new();
        for x in 0..3 {
            let lat = match x {
                0 => rect.min_lat,
                1 => self.case.quantize_lat((rect.min_lat + rect.max_lat) / 2.0),
                _ => rect.max_lat,
            };
            for y in 0..3 {
                let lon = match y {
                    0 => rect.min_lon,
                    1 if x == 1 => continue,
                    1 => self.case.quantize_lon((rect.min_lon + rect.max_lon) / 2.0),
                    _ => rect.max_lon,
                };
                let mut doc = Document::new();
                self.add_point(FIELD_NAME, &mut doc, lat, lon);
                docs.push(doc);
            }
        }
        let index = point_index::build(docs, &[]);
        let count = |r: Rect| point_index::hits(&index, self.rect_query(FIELD_NAME, &r)).len();
        assert_eq!(count(rect), 8, "exact edges of {rect:?}");
        #[allow(clippy::float_cmp)]
        {
            if rect.min_lat != -90.0 {
                let min_lat = rect.min_lat.next_down();
                assert_eq!(count(Rect { min_lat, ..rect }), 8);
            }
            if rect.max_lat != 90.0 {
                let max_lat = rect.max_lat.next_up();
                assert_eq!(count(Rect { max_lat, ..rect }), 8);
            }
            if rect.min_lon != -180.0 {
                let min_lon = rect.min_lon.next_down();
                assert_eq!(count(Rect { min_lon, ..rect }), 8);
            }
            if rect.max_lon != 180.0 {
                let max_lon = rect.max_lon.next_up();
                assert_eq!(count(Rect { max_lon, ..rect }), 8);
            }
            // Java compares minLon with 80, not 180: kept.
            if rect.min_lat != 90.0
                && rect.max_lat != -90.0
                && rect.min_lon != 80.0
                && rect.max_lon != -180.0
                && rect.min_lon != rect.max_lon
            {
                let shrunk = Rect {
                    min_lat: rect.min_lat.next_up(),
                    max_lat: rect.max_lat.next_down(),
                    min_lon: rect.min_lon.next_up(),
                    max_lon: rect.max_lon.next_down(),
                };
                assert_eq!(count(shrunk), 0, "shrunk {shrunk:?}");
            }
        }
    }

    /// `testEquals`: the box, distance and polygon queries equal their
    /// twins and differ from the same query on another field (a box that
    /// is `MatchNoDocsQuery` excepted).
    pub fn equals(&mut self) {
        let rect = geo_test_util::next_box(self.random);
        let q1 = self.rect_query("field", &rect);
        let q2 = self.rect_query("field", &rect);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");
        if !point_index::is_match_no_docs(&q1) {
            let other = self.rect_query("field2", &rect);
            assert!(
                !point_index::query_equals(&q1, &other),
                "{q1:?} == {other:?}"
            );
        }

        let lat = geo_test_util::next_latitude(self.random);
        let lon = geo_test_util::next_longitude(self.random);
        let q1 = self.distance_query("field", lat, lon, 10000.0);
        let q2 = self.distance_query("field", lat, lon, 10000.0);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");
        let other = self.distance_query("field2", lat, lon, 10000.0);
        assert!(
            !point_index::query_equals(&q1, &other),
            "{q1:?} == {other:?}"
        );

        let lats = [
            rect.min_lat,
            rect.max_lat,
            rect.max_lat,
            rect.min_lat,
            rect.min_lat,
        ];
        let lons = [
            rect.min_lon,
            rect.min_lon,
            rect.max_lon,
            rect.max_lon,
            rect.min_lon,
        ];
        let polygon = || Polygon::new(&lats, &lons, Vec::new()).expect("a box's polygon");
        let q1 = self.polygon_query("field", &[polygon()]);
        let q2 = self.polygon_query("field", &[polygon()]);
        assert!(point_index::query_equals(&q1, &q2), "{q1:?} != {q2:?}");
        let other = self.polygon_query("field2", &[polygon()]);
        assert!(
            !point_index::query_equals(&q1, &other),
            "{q1:?} == {other:?}"
        );
    }

    /// `testRandomDistance`: a few ten-document indexes, a hundred
    /// circles each.
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
    /// haversine distance to each document's quantized point.
    pub fn do_random_distance_test(&mut self, num_docs: usize, num_queries: usize) {
        // Java's points-per-leaf, drawn for its `FilterCodec`.
        let _points_in_leaf = 2 + self.random.next_int_bounded(4);
        let mut points = Vec::with_capacity(num_docs);
        let mut docs = Vec::with_capacity(num_docs);
        for _ in 0..num_docs {
            let lat = self
                .case
                .quantize_lat(geo_test_util::next_latitude(self.random));
            let lon = self
                .case
                .quantize_lon(geo_test_util::next_longitude(self.random));
            let mut doc = Document::new();
            self.add_point("field", &mut doc, lat, lon);
            docs.push(doc);
            points.push((lat, lon));
        }
        let index = point_index::build(docs, &[]);
        for _ in 0..num_queries {
            let lat = geo_test_util::next_latitude(self.random);
            let lon = geo_test_util::next_longitude(self.random);
            let radius = 50_000_000.0 * self.random.next_double();
            let expected: BTreeSet<usize> = (0..num_docs)
                .filter(|&d| haversin_meters(lat, lon, points[d].0, points[d].1) <= radius)
                .collect();
            let actual = point_index::hits(&index, self.distance_query("field", lat, lon, radius));
            assert_eq!(actual, expected, "center: ({lat},{lon}), radius={radius}");
        }
    }

    /// One document with the point `(18.313694, -65.227444)` in `field`.
    fn one_point_doc(&self) -> Vec<Document> {
        let mut doc = Document::new();
        self.add_point("field", &mut doc, 18.313_694, -65.227_444);
        vec![doc]
    }

    /// `testBoxBasics`, `testDistanceBasics`, `testPolygonBasics`,
    /// `testPolygonHole`, `testPolygonHoleExcludes` and
    /// `testMultiPolygonBasics`: one point, found (or excluded by a hole).
    pub fn basics(&mut self) {
        let square = |lats: [f64; 2], lons: [f64; 2], holes: Vec<Polygon>| {
            Polygon::new(
                &[lats[0], lats[0], lats[1], lats[1], lats[0]],
                &[lons[0], lons[1], lons[1], lons[0], lons[0]],
                holes,
            )
            .expect("a valid polygon")
        };
        let rect = Rect {
            min_lat: 18.0,
            max_lat: 19.0,
            min_lon: -66.0,
            max_lon: -65.0,
        };
        let outer = |hole: Option<Polygon>| {
            square([18.0, 19.0], [-66.0, -65.0], hole.into_iter().collect())
        };
        let cases = [
            ("box", self.rect_query("field", &rect), 1),
            (
                "distance",
                self.distance_query("field", 18.0, -65.0, 50_000.0),
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
            assert_eq!(point_index::count(self.one_point_doc(), q), want, "{what}");
        }
    }

    /// `testIndexExtremeValues`, `testIndexOutOfRangeValues`,
    /// `testIndexNaNValues` and `testIndexInfValues`.
    pub fn index_values(&mut self) {
        let mut doc = Document::new();
        for (lat, lon) in [
            (90.0, 180.0),
            (90.0, -180.0),
            (-90.0, 180.0),
            (-90.0, -180.0),
        ] {
            self.add_point("foo", &mut doc, lat, lon);
        }
        let lat = "invalid latitude";
        let lon = "invalid longitude";
        for (la, lo, want) in [
            (90f64.next_up(), 50.0, lat),
            ((-90f64).next_down(), 50.0, lat),
            (90.0, 180f64.next_up(), lon),
            (90.0, (-180f64).next_down(), lon),
            (f64::NAN, 50.0, lat),
            (50.0, f64::NAN, lon),
            (f64::INFINITY, 50.0, lat),
            (f64::NEG_INFINITY, 50.0, lat),
            (50.0, f64::INFINITY, lon),
            (50.0, f64::NEG_INFINITY, lon),
        ] {
            let result = self.case.add_point_to_doc("foo", &mut doc, la, lo);
            point_index::assert_rejected(result, &[want]);
        }
    }

    /// `testBoxInvalidCoordinates`, `testDistanceIllegal`,
    /// `testDistanceNegative`, `testDistanceNaN` and `testDistanceInf`.
    pub fn invalid_queries(&mut self) {
        let bad_box = Rect {
            min_lat: -92.0,
            max_lat: -91.0,
            min_lon: 179.0,
            max_lon: 181.0,
        };
        point_index::assert_rejected(self.case.new_rect_query("field", &bad_box), &[]);
        point_index::assert_rejected(
            self.case
                .new_distance_query("field", 92.0, 181.0, 120_000.0),
            &[],
        );
        for radius in [-1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            point_index::assert_rejected(
                self.case.new_distance_query("field", 18.0, 19.0, radius),
                &["radiusMeters", "invalid"],
            );
        }
    }

    /// `searchSmallSet(query, size)`'s hit count: sixteen points in
    /// `"point"`, eight documents holding two of them, and up to nine
    /// documents without.
    fn search_small_set(&mut self, query: Clause) -> usize {
        const PTS: [[f64; 2]; 16] = [
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
        let mut docs = Vec::new();
        for p in PTS {
            let mut doc = Document::new();
            self.add_point("point", &mut doc, p[0], p[1]);
            docs.push(doc);
        }
        for pair in PTS.chunks(2) {
            let mut doc = Document::new();
            for p in pair {
                self.add_point("point", &mut doc, p[0], p[1]);
            }
            docs.push(doc);
        }
        point_index::small_set_string_docs(self.random, &mut docs);
        point_index::count(docs, query)
    }

    /// The `testSmallSet*` family: boxes (one across the dateline, one
    /// the whole map), polygons and circles over [`Self::search_small_set`].
    pub fn small_set(&mut self) {
        let world = Rect {
            min_lat: GeoUtils::MIN_LAT_INCL,
            max_lat: GeoUtils::MAX_LAT_INCL,
            min_lon: GeoUtils::MIN_LON_INCL,
            max_lon: GeoUtils::MAX_LON_INCL,
        };
        let poly = Polygon::new(
            &[
                33.073_130,
                32.994_266_9,
                32.938_386,
                33.037_449_4,
                33.136_976_2,
                33.116_274_7,
                33.073_130,
                33.073_130,
            ],
            &[
                -96.768_264_7,
                -96.828_002_9,
                -96.628_875_7,
                -96.492_919_9,
                -96.604_156_4,
                -96.744_918_8,
                -96.768_264_77,
                -96.768_264_7,
            ],
            vec![],
        )
        .expect("a valid polygon");
        let whole = Polygon::new(
            &[
                world.min_lat,
                world.max_lat,
                world.max_lat,
                world.min_lat,
                world.min_lat,
            ],
            &[
                world.min_lon,
                world.min_lon,
                world.max_lon,
                world.max_lon,
                world.min_lon,
            ],
            vec![],
        )
        .expect("a valid polygon");
        let rect = |min_lat, max_lat, min_lon, max_lon| Rect {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        };
        let cases: Vec<(&str, Clause, usize)> = vec![
            (
                "Rect",
                self.rect_query("point", &rect(32.778, 32.779, -96.778, -96.777)),
                4,
            ),
            (
                "Dateline",
                self.rect_query("point", &rect(-45.0, -44.0, 179.0, -179.0)),
                2,
            ),
            // 3 single-valued docs + 2 multi-valued docs.
            (
                "MultiValued",
                self.rect_query("point", &rect(32.755, 32.776, -96.454, -96.770)),
                5,
            ),
            ("WholeMap", self.rect_query("point", &world), 24),
            ("Poly", self.polygon_query("point", &[poly]), 2),
            ("PolyWholeMap", self.polygon_query("point", &[whole]), 24),
            (
                "Distance",
                self.distance_query(
                    "point",
                    32.948_235_888_393_68,
                    -96.453_811_302_781_1,
                    6000.0,
                ),
                2,
            ),
            (
                "TinyDistance",
                self.distance_query("point", 40.720_611, -73.998_776, 1.0),
                2,
            ),
            // LUCENE-6905.
            (
                "DistanceNotEmpty",
                self.distance_query(
                    "point",
                    -88.560_293_717_309_83,
                    -177.235_376_760_363_58,
                    7_757.999_232_959_935,
                ),
                2,
            ),
            (
                "HugeDistance",
                self.distance_query(
                    "point",
                    32.948_235_888_393_68,
                    -96.453_811_302_781_1,
                    6_000_000.0,
                ),
                16,
            ),
            (
                "DistanceDateline",
                self.distance_query(
                    "point",
                    32.948_235_888_393_68,
                    -179.953_811_302_781_1,
                    120_000.0,
                ),
                3,
            ),
        ];
        for (what, q, want) in cases {
            assert_eq!(self.search_small_set(q), want, "testSmallSet{what}");
        }
    }

    /// `testNarrowPolygonCloseToNorthPole`: a polygon a few quanta wide
    /// at the top of the encoding finds the point inside it.
    pub fn narrow_polygon_close_to_north_pole(&mut self) {
        let base = i32::MAX;
        let mut doc = Document::new();
        self.add_point(
            FIELD_NAME,
            &mut doc,
            GeoEncodingUtils::decode_latitude(base - 2),
            GeoEncodingUtils::decode_longitude(base - 2),
        );
        let min_lat = GeoEncodingUtils::decode_latitude(base - 3);
        let max_lat = GeoEncodingUtils::decode_latitude(base);
        let min_lon = GeoEncodingUtils::decode_longitude(base - 3);
        let max_lon = GeoEncodingUtils::decode_longitude(base);
        let polygon = Polygon::new(
            &[min_lat, min_lat, max_lat, max_lat, min_lat],
            &[min_lon, max_lon, max_lon, min_lon, min_lon],
            vec![],
        )
        .expect("a valid polygon");
        let q = self.polygon_query(FIELD_NAME, &[polygon]);
        assert_eq!(point_index::count(vec![doc], q), 1);
    }
}
