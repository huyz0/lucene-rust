//! `org.apache.lucene.sandbox.search`: the sandbox's queries.

mod doc_values_multi_range;
mod large_num_hits;
mod multi_range_query;
mod profiler_collector;

pub use lucene_search::extended_query::CombinedFieldQuery;
pub use lucene_search::sandbox::search::CoveringQuery;

pub use doc_values_multi_range::{
    ByteRange, LongRange, SortedNumericDocValuesMultiRangeQuery, SortedNumericStabbingBuilder,
    SortedSetDocValuesMultiRangeQuery, SortedSetStabbingBuilder,
};
pub use large_num_hits::{LargeNumHitsTopDocsCollector, LargeNumHitsTopDocsCollectorManager};
pub use lucene_search::sandbox::search::{
    GraphToken, TermAutomatonQuery, TokenStreamToTermAutomatonQuery,
};
pub use lucene_search::sandbox::search::{
    PhraseWildcardBuilder, PhraseWildcardQuery, PhraseWildcardTerm, ResolvedPhraseWildcard,
};
pub use multi_range_query::{MultiRangeQuery, MultiRangeQueryBuilder, RangeClause};
pub use profiler_collector::{
    ProfilerCollector, ProfilerCollectorManager, ProfilerCollectorResult, ProfilerHandle,
};

/// `ConstantScoreScorer.setMinCompetitiveScore(minScore)` with `minScore >
/// score`: the collector's queue is full and its count past the threshold,
/// so no later document at this constant score can enter it, and Java's
/// scorer ends there. The sandbox's constant-score queries stop collecting
/// (or skip a segment) at the same point.
pub(crate) use lucene_search::document::dense_bulk::saturated;
