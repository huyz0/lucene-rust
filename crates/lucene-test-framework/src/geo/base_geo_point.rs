//! `BaseGeoPointTestCase`: random points -- some missing, some sharing a
//! latitude, a longitude or the whole point with an earlier document --
//! indexed by a subclass's field, then queried by random boxes and
//! circles, every hit checked against `rectContainsPoint` or
//! `haversinMeters`.
//!
//! As in [`crate::search::base_range_field`], each verify writes one
//! segment under a plain `IndexWriter`, deletes in one last commit, and
//! draws and ignores the `forceMerge` coin. The polygon and geometry
//! verifies, and the fixed-input tests (`testBoxBasics`, `testSmallSet*`,
//! the invalid-argument checks), are not ported (see the ledger).

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_search::collector::ScoreMode;
use lucene_search::query::Clause;
use lucene_util::geo::GeoUtils;
use lucene_util::java_random::JavaRandom;
use lucene_util::sloppy_math::haversin_meters;

use super::geo_test_util::{self, Rect};
use crate::search::base_explanation::top_level;
use crate::search::check_hits;
use crate::search::test_index::TestIndex;
use crate::util::{lucene_test_case, test_util};

/// `BaseGeoPointTestCase.FIELD_NAME`.
pub const FIELD_NAME: &str = "point";

/// A subclass: the field it indexes and the queries it answers with.
pub trait GeoPointCase {
    /// `addPointToDoc(field, doc, lat, lon)`.
    fn add_point_to_doc(&self, field: &str, doc: &mut Document, lat: f64, lon: f64);
    /// `newRectQuery(field, minLat, maxLat, minLon, maxLon)`.
    fn new_rect_query(&self, field: &str, rect: &Rect) -> Clause;
    /// `newDistanceQuery(field, centerLat, centerLon, radiusMeters)`.
    fn new_distance_query(&self, field: &str, lat: f64, lon: f64, radius_meters: f64) -> Clause;
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
        let n = self.at_least(1000);
        let lat = geo_test_util::next_latitude(self.random);
        let lon = geo_test_util::next_longitude(self.random);
        self.verify(vec![lat; n], vec![lon; n]);
    }

    /// `testLowCardinality`: a tricky adversary for the BKD tree.
    pub fn low_cardinality(&mut self) {
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
        self.verify(lats, lons);
    }

    /// `testAllLatEqual`.
    pub fn all_lat_equal(&mut self) {
        self.all_one_equal(true);
    }

    /// `testAllLonEqual`.
    pub fn all_lon_equal(&mut self) {
        self.all_one_equal(false);
    }

    /// `testAllLatEqual` (`lat_fixed`) or `testAllLonEqual`: some docs
    /// missing, some copying an earlier doc's free coordinate.
    fn all_one_equal(&mut self, lat_fixed: bool) {
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
        self.verify(lats, lons);
    }

    /// `doTestRandom(count)`.
    pub fn do_test_random(&mut self, count: i32) {
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
        self.verify(lats, lons);
    }

    /// `verify(lats, lons)`: quantize as the index does, then the
    /// rectangle and distance verifies.
    pub fn verify(&mut self, mut lats: Vec<f64>, mut lons: Vec<f64>) {
        for (lat, lon) in lats.iter_mut().zip(lons.iter_mut()) {
            if !lat.is_nan() {
                *lat = self.case.quantize_lat(*lat);
                *lon = self.case.quantize_lon(*lon);
            }
        }
        self.verify_random_rectangles(&lats, &lons);
        self.verify_random_distances(&lats, &lons);
    }

    /// `indexPoints`: one document per id (the `id` field as a term and a
    /// doc value), deletes drawn as Java draws them.
    fn index_points(&mut self, lats: &[f64], lons: &[f64]) -> (TestIndex, BTreeSet<usize>) {
        let mut deleted = BTreeSet::new();
        let mut written = Vec::with_capacity(lats.len());
        for id in 0..lats.len() {
            let mut doc = Document::new();
            doc.add(StringField::new("id", id.to_string(), Store::No));
            doc.add(NumericDocValuesField::new(
                "id",
                i64::try_from(id).expect("small"),
            ));
            if !lats[id].is_nan() {
                self.case
                    .add_point_to_doc(FIELD_NAME, &mut doc, lats[id], lons[id]);
            }
            written.push(doc);
            if id > 0 && self.random.next_int_bounded(100) == 42 {
                deleted.insert(self.next_index(id));
            }
        }
        let _force_merge = self.random.next_boolean();
        let ids: Vec<String> = deleted.iter().map(ToString::to_string).collect();
        let deletes: Vec<(&str, &str)> = ids.iter().map(|id| ("id", id.as_str())).collect();
        let index = TestIndex::build(
            Arc::new(Analyzer::standard(None)),
            &[written],
            &deletes,
            &[],
        );
        (index, deleted)
    }

    /// Checks one query's hits against `expected(id)` for every live,
    /// non-missing id; one segment, so doc id == id.
    fn check(
        index: &TestIndex,
        deleted: &BTreeSet<usize>,
        lats: &[f64],
        q: Clause,
        what: &str,
        expected: impl Fn(usize) -> bool,
    ) {
        let q = top_level(q);
        let hits: BTreeSet<usize> = index.with_searcher(|s| {
            check_hits::all_hits(s, &q, ScoreMode::CompleteNoScores)
                .expect("search")
                .into_iter()
                .map(|(doc, _)| usize::try_from(doc).expect("a doc id"))
                .collect()
        });
        let wrong: Vec<usize> = (0..lats.len())
            .filter(|&id| {
                let want = !deleted.contains(&id) && !lats[id].is_nan() && expected(id);
                hits.contains(&id) != want
            })
            .collect();
        assert!(
            wrong.is_empty(),
            "some hits were wrong: ids {wrong:?} for {what}"
        );
    }

    /// `verifyRandomRectangles`.
    fn verify_random_rectangles(&mut self, lats: &[f64], lons: &[f64]) {
        let (index, deleted) = self.index_points(lats, lons);
        let iters = self.at_least(25);
        for _ in 0..iters {
            let rect = geo_test_util::next_box(self.random);
            let q = self.case.new_rect_query(FIELD_NAME, &rect);
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
            let q = self.case.new_distance_query(FIELD_NAME, lat, lon, radius);
            let what = format!("centerLat={lat} centerLon={lon} radiusMeters={radius}");
            Self::check(&index, &deleted, lats, q, &what, |id| {
                haversin_meters(lat, lon, lats[id], lons[id]) <= radius
            });
        }
    }
}
