//! `org.apache.lucene.sandbox.search`: the sandbox's queries.

mod doc_values_multi_range;
mod multi_range_query;

pub use doc_values_multi_range::{
    ByteRange, LongRange, SortedNumericDocValuesMultiRangeQuery, SortedNumericStabbingBuilder,
    SortedSetDocValuesMultiRangeQuery, SortedSetStabbingBuilder,
};
pub use multi_range_query::{MultiRangeQuery, MultiRangeQueryBuilder, RangeClause};

use crate::collector::ScoringCollector;

/// `ConstantScoreScorer.setMinCompetitiveScore(minScore)` with `minScore >
/// score`: the collector's queue is full and its count past the threshold,
/// so no later document at this constant score can enter it, and Java's
/// scorer ends there. The sandbox's constant-score queries stop collecting
/// (or skip a segment) at the same point.
pub(crate) fn saturated(collector: &dyn ScoringCollector, score: f32) -> bool {
    crate::bulk_scorer::min_competitive_score(collector) > score
}
