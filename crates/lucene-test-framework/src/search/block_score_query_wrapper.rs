//! Lucene 10.5.0 `BlockScoreQueryWrapper`: reduce max-score block lengths.
//!
//! The operational scorer lives behind lucene-search's feature-only test seam:
//! it buffers the wrapped scorer's exact docs and scores between -1 and
//! NO_MORE_DOCS sentinels, then uses Java's array-position block boundaries.
//! Scoring modes buffer eagerly; no-score modes return the wrapped scorer.
//! Rewrite, visits, display, explanations and cacheability delegate to the child.
//! Rust uses owned exact boxed slices and `Result` instead of Java arrays and exceptions;
//! query equality is structural (this port has no Query.hashCode contract).

use lucene_search::query::Clause;
use lucene_search::test_support::TestQuery;

/// Java: `new BlockScoreQueryWrapper(query, blockLength)`.
///
/// Like Java, accepts every i32 length. Zero or negative lengths may panic
/// when a scored caller asks for a block boundary; no-score execution bypasses
/// the buffered scorer and its block arithmetic.
pub fn wrap(query: impl Into<Clause>, block_length: i32) -> Clause {
    TestQuery::block_score(query, block_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_search::query::TermQuery;

    #[test]
    fn equality_includes_child_and_length_without_constructor_validation() {
        let a = || TermQuery::new("f", "a");
        assert_eq!(wrap(a(), 3), wrap(a(), 3));
        assert_ne!(wrap(a(), 3), wrap(a(), 4));
        assert_ne!(wrap(a(), 3), wrap(TermQuery::new("f", "b"), 3));
        for length in [i32::MIN, -1, 0, i32::MAX] {
            assert_eq!(wrap(a(), length), wrap(a(), length));
        }
    }
}
