//! `LongDistanceFeatureQuery` (`LongField.newDistanceFeatureQuery`): every
//! document with a value for a `long` field, scored by how close the value is
//! to an origin -- `weight * pivot / (pivot + |value - origin|)` -- with the
//! field's points used to skip the documents that can no longer compete once
//! the collector has a threshold.
//!
//! Ported from `LongDistanceFeatureQuery.DistanceScorer` as
//! `DefaultBulkScorer` drives it: the scorer iterates the documents with
//! doc values; whenever the collector's minimum competitive score rises, it
//! computes the largest distance that can still reach it and, when the
//! points say few enough documents lie within it, replaces its iterator by
//! those documents (after the current one).

use lucene_codecs::doc_values::{self, NumericEntry, SortedNumericEntry};
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::points::{IntersectVisitor, PointsReader, Relation};
use lucene_index::document::sortable_bytes_to_long;

use super::{field_info, reader, DocumentQuery};
use crate::collector::ScoringCollector;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

const MIN_SKIP_INTERVAL: i32 = 32;
const MAX_SKIP_INTERVAL: i32 = 8192;

/// `LongDistanceFeatureQuery`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LongDistanceFeatureQuery {
    pub field: String,
    pub origin: i64,
    pub pivot_distance: i64,
}

impl LongDistanceFeatureQuery {
    /// `LongDistanceFeatureQuery(field, origin, pivotDistance)`: the pivot
    /// must be positive.
    pub fn new(field: impl Into<String>, origin: i64, pivot_distance: i64) -> Result<Self> {
        if pivot_distance <= 0 {
            return Err(Error::DocumentQuery(format!(
                "pivotDistance must be > 0, got {pivot_distance}"
            )));
        }
        Ok(LongDistanceFeatureQuery {
            field: field.into(),
            origin,
            pivot_distance,
        })
    }

    /// `DistanceScorer.score(distance)`.
    fn score(&self, boost: f32, distance: i64) -> f32 {
        let pivot = self.pivot_distance as f64;
        (f64::from(boost) * (pivot / (pivot + distance as f64))) as f32
    }

    /// The distance of `value` from the origin, `Long.MAX_VALUE` when it
    /// overflows.
    fn distance(&self, value: i64) -> i64 {
        let d = value.max(self.origin).wrapping_sub(value.min(self.origin));
        if d < 0 {
            i64::MAX
        } else {
            d
        }
    }

    /// `selectValue`: of a document's ascending values, the one closest to
    /// the origin (the lower on a tie).
    fn select_value(&self, values: &[i64]) -> Option<i64> {
        let first = *values.first()?;
        if values.len() == 1 || first >= self.origin {
            return Some(first);
        }
        let mut previous = first;
        for &next in &values[1..] {
            if next >= self.origin {
                let below = (self.origin.wrapping_sub(previous)) as u64;
                let above = (next.wrapping_sub(self.origin)) as u64;
                return Some(if below < above { previous } else { next });
            }
            previous = next;
        }
        Some(previous)
    }

    /// `computeMaxDistance(minScore, previousMaxDistance)`: the largest
    /// distance still scoring at least `min_score`.
    fn compute_max_distance(&self, boost: f32, min_score: f32, previous: i64) -> i64 {
        if self.score(boost, previous) >= min_score {
            return previous;
        }
        let (mut min, mut max) = (0i64, previous);
        while max.wrapping_sub(min) > 1 {
            let mid = ((min as u64).wrapping_add(max as u64) >> 1) as i64;
            if self.score(boost, mid) >= min_score {
                min = mid;
            } else {
                max = mid;
            }
        }
        min
    }
}

/// The field's values, singleton or multi-valued.
enum Values<'a> {
    Numeric(&'a NumericEntry),
    Sorted(&'a SortedNumericEntry),
}

/// The documents the scorer still iterates.
enum Candidates {
    /// Every document with a value.
    All,
    /// These, ascending, from this index on (`DocIdSetBuilder`'s set).
    List(Vec<i32>, usize),
}

/// The `IntersectVisitor` of `setMinCompetitiveScore`: documents after `doc`
/// with a point in `[min, max]`.
struct NearVisitor {
    doc: i32,
    min: i64,
    max: i64,
    docs: Vec<i32>,
}

impl IntersectVisitor for NearVisitor {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        let lo = sortable_bytes_to_long(min_packed);
        let hi = sortable_bytes_to_long(max_packed);
        if lo > self.max || hi < self.min {
            Relation::CellOutsideQuery
        } else if lo < self.min || hi > self.max {
            Relation::CellCrossesQuery
        } else {
            Relation::CellInsideQuery
        }
    }

    fn visit(&mut self, doc_id: i32) {
        if doc_id > self.doc {
            self.docs.push(doc_id);
        }
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if doc_id <= self.doc {
            return;
        }
        let v = sortable_bytes_to_long(packed_value);
        if v >= self.min && v <= self.max {
            self.docs.push(doc_id);
        }
    }
}

/// `DistanceScorer`'s pruning state.
struct Pruning {
    max_distance: i64,
    current_skip_interval: i32,
    try_update_fail_count: i32,
    set_min_competitive_score_counter: i32,
    last_min: Option<f32>,
}

impl Pruning {
    /// `updateSkipInterval(success)`.
    fn update_skip_interval(&mut self, success: bool) {
        if self.set_min_competitive_score_counter > 256 {
            if success {
                self.current_skip_interval =
                    (self.current_skip_interval / 2).max(MIN_SKIP_INTERVAL);
                self.try_update_fail_count = 0;
            } else if self.try_update_fail_count >= 3 {
                self.current_skip_interval = self
                    .current_skip_interval
                    .saturating_mul(2)
                    .min(MAX_SKIP_INTERVAL);
                self.try_update_fail_count = 0;
            } else {
                self.try_update_fail_count = self.try_update_fail_count.saturating_add(1);
            }
        }
    }
}

impl DocumentQuery for LongDistanceFeatureQuery {
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
        let (data, values) = match info.doc_values_type {
            DocValuesType::None => return Ok(()),
            DocValuesType::Numeric | DocValuesType::SortedNumeric => {
                let Some((meta, data)) = r.doc_values_for_field(info.number) else {
                    return Ok(());
                };
                if let Some(e) = meta.numeric_entry(info.number) {
                    (data, Values::Numeric(e))
                } else if let Some(e) = meta.sorted_numeric_entry(info.number) {
                    (data, Values::Sorted(e))
                } else {
                    return Ok(());
                }
            }
            other => {
                return Err(Error::DocumentQuery(format!(
                    "unexpected docvalues type {} for field '{}' (expected one of \
                     [SORTED_NUMERIC, NUMERIC]). Re-index with correct docvalues type.",
                    lucene_index::document::doc_values_type_name(other),
                    self.field
                )))
            }
        };
        let value_of = |doc: i32| -> Result<Option<i64>> {
            Ok(match &values {
                Values::Numeric(e) => doc_values::numeric_value(data, e, doc)?,
                Values::Sorted(e) => {
                    self.select_value(&doc_values::sorted_numeric_values(data, e, doc)?)
                }
            })
        };
        // `docValues.cost()`: the documents with a value -- the lead cost of
        // a top-level query.
        let mut with_value = Vec::new();
        for doc in 0..r.max_doc {
            if value_of(doc)?.is_some() {
                with_value.push(doc);
            }
        }
        let lead_cost = with_value.len() as i64;
        let mut candidates = Candidates::All;
        let mut state = Pruning {
            max_distance: i64::MAX,
            current_skip_interval: MIN_SKIP_INTERVAL,
            try_update_fail_count: 0,
            set_min_competitive_score_counter: 0,
            last_min: None,
        };
        let mut all_at = 0usize;
        loop {
            let doc = match &mut candidates {
                Candidates::All => match with_value.get(all_at) {
                    Some(&d) => {
                        all_at = all_at.saturating_add(1);
                        d
                    }
                    None => break,
                },
                Candidates::List(docs, at) => match docs.get(*at) {
                    Some(&d) => {
                        *at = at.saturating_add(1);
                        d
                    }
                    None => break,
                },
            };
            if !leaf.live_docs.is_none_or(|bits| bits.get_doc(doc)) {
                continue;
            }
            // `DistanceScorer.score()`: a candidate without a value scores 0.
            let score = match value_of(doc)? {
                Some(v) => self.score(boost, self.distance(v)),
                None => 0.0,
            };
            collector.collect(doc, score);
            let Some(min_score) = collector.pruning_threshold() else {
                continue;
            };
            if state.last_min.is_some_and(|m| m >= min_score) {
                continue;
            }
            state.last_min = Some(min_score);
            // `setMinCompetitiveScore(minScore)`.
            if min_score > boost {
                break;
            }
            state.set_min_competitive_score_counter =
                state.set_min_competitive_score_counter.saturating_add(1);
            let interval = state.current_skip_interval;
            if state.set_min_competitive_score_counter > 256
                && (state.set_min_competitive_score_counter & (interval - 1)) != interval - 1
            {
                continue;
            }
            let previous = state.max_distance;
            state.max_distance = self.compute_max_distance(boost, min_score, previous);
            if state.max_distance == previous {
                continue;
            }
            let mut min_value = self.origin.wrapping_sub(state.max_distance);
            if min_value > self.origin {
                min_value = i64::MIN;
            }
            let mut max_value = self.origin.wrapping_add(state.max_distance);
            if max_value < self.origin {
                max_value = i64::MAX;
            }
            let mut visitor = NearVisitor {
                doc,
                min: min_value,
                max: max_value,
                docs: Vec::new(),
            };
            let it_cost = match &candidates {
                Candidates::All => lead_cost,
                Candidates::List(docs, _) => docs.len() as i64,
            };
            let threshold = lead_cost.min(it_cost) >> 3;
            let estimate =
                points.estimate_point_count_bounded(info.number, &mut visitor, threshold)?;
            if estimate >= threshold {
                state.update_skip_interval(false);
                continue;
            }
            points.intersect(info.number, &mut visitor)?;
            let mut docs = visitor.docs;
            docs.sort_unstable();
            docs.dedup();
            candidates = Candidates::List(docs, 0);
            state.update_skip_interval(true);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_distances_and_scores() {
        let q = LongDistanceFeatureQuery::new("f", 10, 5).unwrap();
        assert!(LongDistanceFeatureQuery::new("f", 10, 0).is_err());
        assert_eq!(q.select_value(&[]), None);
        assert_eq!(q.select_value(&[3]), Some(3));
        assert_eq!(q.select_value(&[12, 20]), Some(12));
        assert_eq!(
            q.select_value(&[7, 13]),
            Some(13),
            "a tie goes to the upper"
        );
        assert_eq!(q.select_value(&[8, 13]), Some(8));
        assert_eq!(q.select_value(&[7, 12]), Some(12));
        assert_eq!(q.select_value(&[1, 2, 3]), Some(3));
        assert_eq!(q.distance(3), 7);
        assert_eq!(q.distance(i64::MIN), i64::MAX, "overflow saturates");
        assert_eq!(q.score(2.0, 0), 2.0);
        assert_eq!(q.score(1.0, 5), 0.5);
        let d = q.compute_max_distance(1.0, 0.5, i64::MAX);
        assert!(q.score(1.0, d) >= 0.5 && q.score(1.0, d + 1) < 0.5);
        assert_eq!(q.compute_max_distance(1.0, 0.1, 3), 3);
        let mut p = Pruning {
            max_distance: 0,
            current_skip_interval: 64,
            try_update_fail_count: 0,
            set_min_competitive_score_counter: 300,
            last_min: None,
        };
        p.update_skip_interval(true);
        assert_eq!(p.current_skip_interval, 32);
        for _ in 0..4 {
            p.update_skip_interval(false);
        }
        assert_eq!(p.current_skip_interval, 64);
        let mut v = NearVisitor {
            doc: 5,
            min: 0,
            max: 10,
            docs: Vec::new(),
        };
        let b = |x: i64| lucene_index::document::long_to_sortable_bytes(x);
        assert_eq!(v.compare(&b(11), &b(20)), Relation::CellOutsideQuery);
        assert_eq!(v.compare(&b(-1), &b(5)), Relation::CellCrossesQuery);
        assert_eq!(v.compare(&b(1), &b(5)), Relation::CellInsideQuery);
        v.visit(4);
        v.visit(6);
        v.visit_with_value(7, &b(3));
        v.visit_with_value(8, &b(30));
        v.visit_with_value(2, &b(3));
        assert_eq!(v.docs, vec![6, 7]);
    }
}
