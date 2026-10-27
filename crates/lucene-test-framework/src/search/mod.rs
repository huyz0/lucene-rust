//! `org.apache.lucene.tests.search`: checks every query should pass
//! ([`query_utils`], [`check_hits`]), the asserting searcher
//! ([`asserting`], [`asserting_matches`]), the small in-memory index the bases search
//! ([`test_index`]), and the test bases ([`base_explanation`],
//! [`search_equivalence`], [`matches_test_base`], [`base_range_field`]),
//! and the similarity bases ([`similarities`]).

pub mod asserting;
pub mod asserting_matches;
pub mod base_explanation;
pub mod base_range_field;
pub mod block_score_query_wrapper;
pub mod bulk_scorer_wrapper_scorer;
pub mod check_hits;
pub mod disabling_bulk_scorer_query;
pub mod dummy_total_hit_count;
pub mod matches_test_base;
pub mod new_searcher;
pub mod query_utils;
pub mod scorer_index_searcher;
pub mod search_equivalence;
pub mod shard_searching;
pub mod similarities;
pub mod test_index;
