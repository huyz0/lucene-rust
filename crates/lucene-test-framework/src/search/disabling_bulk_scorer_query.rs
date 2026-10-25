//! Lucene's `DisablingBulkScorerQuery`: preserve a query's scorer while
//! selecting `DefaultBulkScorer` instead of a specialized bulk scorer.
//!
//! Java installs a `FilterWeight` and a `ScorerSupplier`; Rust represents the
//! wrapper as a feature-only query value. The existing extended-query bulk
//! path builds its scorer and drives it through `DefaultBulkScorer`. Query
//! rewriting, visiting, explanations, matches and cacheability delegate to
//! the wrapped query. The wrapper remains distinct under query equality.

use lucene_search::query::Clause;
use lucene_search::test_support::TestQuery;

/// Java: `new DisablingBulkScorerQuery(query)`.
///
/// The result can occur anywhere a clause can occur, including within a
/// Boolean query. It disables specialized bulk scoring for this clause.
pub fn wrap(query: impl Into<Clause>) -> Clause {
    TestQuery::disabling_bulk_scorer(query)
}
