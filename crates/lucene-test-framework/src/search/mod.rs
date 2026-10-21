//! `org.apache.lucene.tests.search`: checks every query should pass
//! ([`query_utils`], [`check_hits`]), the asserting searcher
//! ([`asserting`]) and the small in-memory index the checks search
//! ([`test_index`]).

pub mod asserting;
pub mod check_hits;
pub mod query_utils;
pub mod test_index;
