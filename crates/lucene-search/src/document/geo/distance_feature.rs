//! `LatLonPointDistanceFeatureQuery` (`LatLonPoint.newDistanceFeatureQuery`):
//! every document with a `LatLonDocValuesField` value, scored by its
//! distance to an origin -- `weight * pivot / (pivot + distance)` -- with the
//! field's points used to skip the documents that can no longer compete
//! once the collector has a threshold.
//!
//! Ported from `DistanceScorer` as `DefaultBulkScorer` drives it, the way
//! [`crate::document::LongDistanceFeatureQuery`] ports its `long` twin: the
//! scorer walks the documents with doc values; whenever the collector's
//! minimum competitive score rises (every time for the first 256 rises,
//! then every 32nd), it computes the largest distance that still reaches
//! it, and when the points estimate few enough documents inside that
//! distance's bounding box, it replaces its iterator by those documents
//! (after the current one).

use lucene_codecs::points::{IntersectVisitor, PointsReader, Relation};
use lucene_index::document::{doc_value_high, doc_value_low, sortable_bytes_to_int};
use lucene_util::geo::{GeoEncodingUtils, GeoUtils, Rectangle};
use lucene_util::sloppy_math;

use super::{geo, illegal, sorted_numeric};
use crate::collector::ScoringCollector;
use crate::document::distance_feature::BuilderCost;
use crate::document::{field_info, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::Result;

/// `LatLonPointDistanceFeatureQuery`.
#[derive(Debug, Clone, PartialEq)]
pub struct LatLonPointDistanceFeatureQuery {
    pub field: String,
    pub origin_lat: f64,
    pub origin_lon: f64,
    pub pivot_distance: f64,
}

impl LatLonPointDistanceFeatureQuery {
    /// `LatLonPointDistanceFeatureQuery(field, originLat, originLon,
    /// pivotDistance)`.
    ///
    /// # Errors
    /// An invalid origin, or a pivot that is not positive, with Java's
    /// message.
    pub fn new(
        field: impl Into<String>,
        origin_lat: f64,
        origin_lon: f64,
        pivot_distance: f64,
    ) -> Result<Self> {
        GeoUtils::check_latitude(origin_lat).map_err(geo)?;
        GeoUtils::check_longitude(origin_lon).map_err(geo)?;
        // `pivotDistance <= 0` is false for NaN, which Java accepts too.
        if pivot_distance <= 0.0 {
            return Err(illegal(format!(
                "pivotDistance must be > 0, got {}",
                lucene_util::geo::java_double_string(pivot_distance)
            )));
        }
        Ok(LatLonPointDistanceFeatureQuery {
            field: field.into(),
            origin_lat,
            origin_lon,
            pivot_distance,
        })
    }

    /// `getDistanceKeyFromEncoded(encoded)`.
    fn distance_key(&self, encoded: i64) -> f64 {
        sloppy_math::haversin_sort_key(
            self.origin_lat,
            self.origin_lon,
            GeoEncodingUtils::decode_latitude(doc_value_high(encoded)),
            GeoEncodingUtils::decode_longitude(doc_value_low(encoded)),
        )
    }

    /// `selectValue`: of a document's values, the closest (the first of
    /// equally close ones).
    fn select_value(&self, values: &[i64]) -> Option<i64> {
        let (&first, rest) = values.split_first()?;
        if rest.is_empty() {
            return Some(first);
        }
        let mut value = first;
        let mut distance = self.distance_key(first);
        for &next in rest {
            let d = self.distance_key(next);
            if d < distance {
                distance = d;
                value = next;
            }
        }
        Some(value)
    }

    /// `DistanceScorer.score(distance)`.
    fn score(&self, boost: f32, distance: f64) -> f32 {
        (f64::from(boost) * (self.pivot_distance / (self.pivot_distance + distance))) as f32
    }

    /// `computeMaxDistance(minScore, previousMaxDistance)`.
    fn compute_max_distance(&self, boost: f32, min_score: f32, previous: f64) -> f64 {
        if self.score(boost, previous) >= min_score {
            return previous;
        }
        let (mut min, mut max) = (0.0f64, previous);
        while max - min > 1.0 {
            let mid = (min + max) / 2.0;
            if self.score(boost, mid) >= min_score {
                min = mid;
            } else {
                max = mid;
            }
        }
        min
    }
}

/// The documents the scorer still iterates.
enum Candidates {
    /// Every document with a value.
    All,
    /// These, ascending, from this index on (`DocIdSetBuilder`'s set), and
    /// that set's cost.
    List(Vec<i32>, usize, i64),
}

/// The `IntersectVisitor` of `setMinCompetitiveScore`: documents after
/// `doc` with a point inside the encoded box.
struct NearVisitor {
    doc: i32,
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
    cross_dateline: bool,
    docs: Vec<i32>,
    builder: BuilderCost,
}

impl NearVisitor {
    fn add(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
        self.builder.add();
    }
}

impl IntersectVisitor for NearVisitor {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let lat_lo = sortable_bytes_to_int(min);
        let lat_hi = sortable_bytes_to_int(max);
        if lat_lo > self.max_lat || lat_hi < self.min_lat {
            return Relation::CellOutsideQuery;
        }
        let mut crosses = lat_lo < self.min_lat || lat_hi > self.max_lat;
        let lon_lo = sortable_bytes_to_int(&min[4..]);
        let lon_hi = sortable_bytes_to_int(&max[4..]);
        if self.cross_dateline {
            if lon_lo > self.max_lon && lon_hi < self.min_lon {
                return Relation::CellOutsideQuery;
            }
            crosses |= lon_lo < self.max_lon || lon_hi > self.min_lon;
        } else {
            if lon_lo > self.max_lon || lon_hi < self.min_lon {
                return Relation::CellOutsideQuery;
            }
            crosses |= lon_lo < self.min_lon || lon_hi > self.max_lon;
        }
        if crosses {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }

    fn visit(&mut self, doc_id: i32) {
        if doc_id > self.doc {
            self.add(doc_id);
        }
    }

    fn grow(&mut self, count: usize) {
        self.builder.grow(i64::try_from(count).unwrap_or(i64::MAX));
    }

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if doc_id <= self.doc {
            return;
        }
        let lat = sortable_bytes_to_int(packed);
        if lat > self.max_lat || lat < self.min_lat {
            return;
        }
        let lon = sortable_bytes_to_int(&packed[4..]);
        if self.cross_dateline {
            if lon < self.min_lon && lon > self.max_lon {
                return;
            }
        } else if lon > self.max_lon || lon < self.min_lon {
            return;
        }
        self.add(doc_id);
    }
}

impl DocumentQuery for LatLonPointDistanceFeatureQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        if info.point_dimension_count == 0 {
            return Ok(());
        }
        let r = reader(leaf)?;
        let points: PointsReader<'_> = r.points_reader()?;
        let Some(pf) = points.field(info.number) else {
            return Ok(());
        };
        super::point_queries::check_points_shape(&info.name, pf)?;
        let Some(mut values) = sorted_numeric(leaf, info)? else {
            // `DocValues.emptySortedNumeric()`: no document to score.
            return Ok(());
        };
        // `docValues.cost()`, also the lead cost of a top-level query. The
        // values themselves are read only for the documents the iterator
        // reaches, as `docValues.nextDoc()` / `advanceExact()` read them.
        let lead_cost = values.cost();
        let mut buf = Vec::new();
        let mut candidates = Candidates::All;
        let mut max_distance = GeoUtils::EARTH_MEAN_RADIUS_METERS * std::f64::consts::PI;
        let mut counter = 0i32;
        // The leaf collector's `minCompetitiveScore`: the last push.
        let mut last_min = 0.0f32;
        // `Candidates::All`'s next document to read.
        let mut next_all = 0i32;
        // `docID()`: -1 before the first document.
        let mut doc = -1;
        loop {
            // `setMinCompetitiveScore(minScore)`, which `TopScoreDocCollector`
            // calls when a leaf starts (`setScorer`) and after a collected
            // document raises its bar: `Math.nextUp` of the worst kept score,
            // since a tie loses to the earlier document -- and only when that
            // is above the leaf's last push, 0 to start with
            // (`localMinScore > minCompetitiveScore`). So a NaN score (a NaN
            // pivot) is never pushed and nothing is pruned.
            'update: {
                let min_score = crate::bulk_scorer::min_competitive_score(collector);
                // Negated so a NaN threshold fails the test, as in Java.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                if !(min_score > last_min) {
                    break 'update;
                }
                last_min = min_score;
                if min_score > boost {
                    // `it = DocIdSetIterator.empty()`.
                    return Ok(());
                }
                counter = counter.saturating_add(1);
                if counter > 256 && (counter & 0x1f) != 0x1f {
                    break 'update;
                }
                let previous = max_distance;
                max_distance = self.compute_max_distance(boost, min_score, previous);
                #[allow(clippy::float_cmp)]
                if max_distance == previous {
                    break 'update;
                }
                let b =
                    Rectangle::from_point_distance(self.origin_lat, self.origin_lon, max_distance)
                        .map_err(geo)?;
                let mut visitor = NearVisitor {
                    doc,
                    min_lat: GeoEncodingUtils::encode_latitude(b.min_lat).map_err(geo)?,
                    max_lat: GeoEncodingUtils::encode_latitude(b.max_lat).map_err(geo)?,
                    min_lon: GeoEncodingUtils::encode_longitude(b.min_lon).map_err(geo)?,
                    max_lon: GeoEncodingUtils::encode_longitude(b.max_lon).map_err(geo)?,
                    cross_dateline: b.crosses_dateline(),
                    docs: Vec::new(),
                    builder: BuilderCost::new(r.max_doc),
                };
                let it_cost = match &candidates {
                    Candidates::All => lead_cost,
                    Candidates::List(_, _, cost) => *cost,
                };
                let threshold = lead_cost.min(it_cost) >> 3;
                let estimate =
                    points.estimate_point_count_bounded(info.number, &mut visitor, threshold)?;
                if estimate >= threshold {
                    break 'update;
                }
                points.intersect(info.number, &mut visitor)?;
                let mut docs = visitor.docs;
                docs.sort_unstable();
                docs.dedup();
                let cost = visitor.builder.cost(docs.len());
                candidates = Candidates::List(docs, 0, cost);
            }
            // `nextDoc()`, skipping deleted documents (`DefaultBulkScorer`
            // with the live docs as accept bits). Every document is read at
            // most once: the candidates ascend.
            let selected = loop {
                doc = match &mut candidates {
                    Candidates::All => {
                        // `docValues.nextDoc()`: the next document with a value.
                        let mut found = None;
                        while next_all < r.max_doc {
                            let d = next_all;
                            next_all = next_all.saturating_add(1);
                            values.values(d, &mut buf)?;
                            if let Some(v) = self.select_value(&buf) {
                                found = Some((d, v));
                                break;
                            }
                        }
                        let Some((d, v)) = found else {
                            return Ok(());
                        };
                        if leaf.live_docs.is_none_or(|bits| bits.get_doc(d)) {
                            doc = d;
                            break Some(v);
                        }
                        continue;
                    }
                    Candidates::List(docs, at, _) => match docs.get(*at) {
                        Some(&d) => {
                            *at = at.saturating_add(1);
                            d
                        }
                        None => return Ok(()),
                    },
                };
                if leaf.live_docs.is_none_or(|bits| bits.get_doc(doc)) {
                    // `score()`: `docValues.advanceExact(docID())`; a
                    // candidate the points found without a doc value scores 0.
                    values.values(doc, &mut buf)?;
                    break self.select_value(&buf);
                }
            };
            let score = match selected {
                Some(v) => self.score(
                    boost,
                    sloppy_math::haversin_meters_from_sort_key(self.distance_key(v)),
                ),
                None => 0.0,
            };
            collector.collect(doc, score);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_index::document::{LatLonDocValuesField, LatLonPoint};

    #[test]
    fn validation_and_scoring() {
        assert!(LatLonPointDistanceFeatureQuery::new("f", 91.0, 0.0, 1.0).is_err());
        assert!(LatLonPointDistanceFeatureQuery::new("f", 0.0, 181.0, 1.0).is_err());
        let e = LatLonPointDistanceFeatureQuery::new("f", 0.0, 0.0, 0.0).unwrap_err();
        assert!(
            e.to_string().contains("pivotDistance must be > 0, got 0.0"),
            "{e}"
        );
        let q = LatLonPointDistanceFeatureQuery::new("f", 0.0, 0.0, 1000.0).unwrap();
        assert_eq!(q.score(2.0, 0.0), 2.0);
        assert_eq!(q.score(1.0, 1000.0), 0.5);
        let d = q.compute_max_distance(1.0, 0.5, 1e7);
        assert!(q.score(1.0, d) >= 0.5 && q.score(1.0, d + 1.0) < 0.5);
        assert_eq!(q.compute_max_distance(1.0, 0.0001, 3.0), 3.0);
        let near = LatLonDocValuesField::encode(0.0, 0.1).unwrap();
        let far = LatLonDocValuesField::encode(0.0, 1.0).unwrap();
        assert_eq!(q.select_value(&[]), None);
        assert_eq!(q.select_value(&[far]), Some(far));
        assert_eq!(q.select_value(&[far, near]), Some(near));
        assert_eq!(q.select_value(&[near, far]), Some(near));
    }

    #[test]
    fn near_visitor_handles_the_dateline() {
        let enc = |lat, lon| LatLonPoint::encode(lat, lon).unwrap();
        let lat = |v| GeoEncodingUtils::encode_latitude(v).unwrap();
        let lon = |v| GeoEncodingUtils::encode_longitude(v).unwrap();
        let mut v = NearVisitor {
            doc: 5,
            min_lat: lat(-10.0),
            max_lat: lat(10.0),
            min_lon: lon(170.0),
            max_lon: lon(-170.0),
            cross_dateline: true,
            docs: Vec::new(),
            builder: BuilderCost::new(1000),
        };
        assert_eq!(
            v.compare(&enc(20.0, 0.0), &enc(30.0, 10.0)),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            v.compare(&enc(-5.0, -100.0), &enc(5.0, 100.0)),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            v.compare(&enc(-5.0, 175.0), &enc(5.0, 179.0)),
            Relation::CellCrossesQuery
        );
        v.visit(3);
        v.visit(6);
        v.visit_with_value(7, &enc(0.0, 175.0));
        v.visit_with_value(8, &enc(0.0, 0.0));
        v.visit_with_value(9, &enc(50.0, 175.0));
        v.visit_with_value(2, &enc(0.0, 175.0));
        assert_eq!(v.docs, vec![6, 7]);
        let mut w = NearVisitor {
            doc: -1,
            min_lat: lat(-10.0),
            max_lat: lat(10.0),
            min_lon: lon(-10.0),
            max_lon: lon(10.0),
            cross_dateline: false,
            docs: Vec::new(),
            builder: BuilderCost::new(1000),
        };
        assert_eq!(
            w.compare(&enc(-5.0, -5.0), &enc(5.0, 5.0)),
            Relation::CellInsideQuery
        );
        assert_eq!(
            w.compare(&enc(-5.0, 20.0), &enc(5.0, 30.0)),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            w.compare(&enc(-5.0, -20.0), &enc(5.0, 5.0)),
            Relation::CellCrossesQuery
        );
        w.visit_with_value(1, &enc(0.0, 0.0));
        w.visit_with_value(2, &enc(0.0, 50.0));
        assert_eq!(w.docs, vec![1]);
    }
}
