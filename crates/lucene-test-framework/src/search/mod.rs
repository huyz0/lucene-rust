//! `org.apache.lucene.tests.search`: checks every query should pass
//! ([`query_utils`], [`check_hits`]), the asserting searcher
//! ([`asserting`], [`asserting_matches`]), the small in-memory index the bases search
//! ([`test_index`]), and the test bases ([`base_explanation`],
//! [`search_equivalence`], [`matches_test_base`]).

pub mod asserting;
pub mod asserting_matches;
pub mod base_explanation;
pub mod check_hits;
pub mod matches_test_base;
pub mod query_utils;
pub mod search_equivalence;
pub mod test_index;
