//! `org.apache.lucene.sandbox.search`: the sandbox's queries.

pub(crate) mod covering;
mod doc_values_multi_range;
mod multi_range_query;
pub(crate) mod phrase_wildcard;
pub(crate) mod term_automaton;

pub use crate::extended_query::CombinedFieldQuery;
pub use covering::CoveringQuery;

pub use doc_values_multi_range::{
    ByteRange, LongRange, SortedNumericDocValuesMultiRangeQuery, SortedNumericStabbingBuilder,
    SortedSetDocValuesMultiRangeQuery, SortedSetStabbingBuilder,
};
pub use multi_range_query::{MultiRangeQuery, MultiRangeQueryBuilder, RangeClause};
pub use phrase_wildcard::{
    PhraseWildcardBuilder, PhraseWildcardQuery, PhraseWildcardTerm, ResolvedPhraseWildcard,
};
pub use term_automaton::{GraphToken, TermAutomatonQuery, TokenStreamToTermAutomatonQuery};

use crate::collector::ScoringCollector;

/// `ConstantScoreScorer.setMinCompetitiveScore(minScore)` with `minScore >
/// score`: the collector's queue is full and its count past the threshold,
/// so no later document at this constant score can enter it, and Java's
/// scorer ends there. The sandbox's constant-score queries stop collecting
/// (or skip a segment) at the same point.
pub(crate) fn saturated(collector: &dyn ScoringCollector, score: f32) -> bool {
    crate::bulk_scorer::min_competitive_score(collector) > score
}
