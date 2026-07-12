//! `TermQuery`-equivalent (`org.apache.lucene.search.TermQuery`), pared down
//! to this slice's scope: a field name plus a single exact term, no scoring
//! metadata attached (`TermQuery` in real Lucene also carries an optional
//! `TermStates` for cross-segment stats reuse — not needed for a
//! single-segment, no-relevance-scoring first cut, see `lib.rs`'s module
//! doc for the full design rationale).

/// A single exact-term lookup against one field, e.g. `TermQuery::new("body",
/// "cat")` — the Rust analogue of `new TermQuery(new Term("body", "cat"))`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TermQuery {
    pub field: String,
    pub term: Vec<u8>,
}

impl TermQuery {
    pub fn new(field: impl Into<String>, term: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            term: term.into(),
        }
    }
}

/// `BooleanQuery`-equivalent (`org.apache.lucene.search.BooleanQuery`), pared down to
/// this slice's scope: a flat list of exact-`TermQuery` clauses per `Occur` bucket
/// (`MUST`, `SHOULD`, `MUST_NOT`) — no nested `BooleanQuery`, no `FILTER` (a `FILTER`
/// clause only differs from `MUST` by not contributing to scoring, and this slice has
/// no scoring yet, so it would be a distinction without a difference here), no
/// `minimumNumberShouldMatch`.
///
/// **Why three flat `Vec<TermQuery>` fields instead of real Lucene's single
/// `Vec<(Occur, Query)>` clause list**: real `BooleanQuery` stores clauses in
/// insertion order because `Occur` is per-clause and clause order matters for some
/// scoring/explain paths. This port has no scoring yet and no nested query types (every
/// clause is a `TermQuery`), so grouping by `Occur` up front removes a dispatch step
/// `search_boolean_query` would otherwise redo on every call (partition-by-`Occur`),
/// with no loss of information this slice actually uses. If nested `BooleanQuery`
/// clauses or scoring-sensitive clause order land later, revisit — the
/// `Vec<(Occur, Query)>` shape earns its keep once clause order or query nesting
/// matters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BooleanQuery {
    /// `Occur.MUST`: every doc must match every clause here (conjunction).
    pub must: Vec<TermQuery>,
    /// `Occur.SHOULD`: a doc must match at least one clause here, but only when
    /// `must` is empty — matching real `BooleanQuery`'s "SHOULD clauses become purely
    /// score-contributing, not filtering, once a MUST/FILTER clause exists" rule (no
    /// `minimumNumberShouldMatch` support yet, so that's the only interaction this
    /// slice implements; see `search_boolean_query`'s doc comment in `lib.rs`).
    pub should: Vec<TermQuery>,
    /// `Occur.MUST_NOT`: a doc must match none of these clauses.
    pub must_not: Vec<TermQuery>,
}

impl BooleanQuery {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_must(mut self, clauses: impl IntoIterator<Item = TermQuery>) -> Self {
        self.must.extend(clauses);
        self
    }

    pub fn with_should(mut self, clauses: impl IntoIterator<Item = TermQuery>) -> Self {
        self.should.extend(clauses);
        self
    }

    pub fn with_must_not(mut self, clauses: impl IntoIterator<Item = TermQuery>) -> Self {
        self.must_not.extend(clauses);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_stores_field_and_term_bytes() {
        let q = TermQuery::new("body", "cat");
        assert_eq!(q.field, "body");
        assert_eq!(q.term, b"cat");
    }

    #[test]
    fn equality_is_field_and_term_based() {
        assert_eq!(TermQuery::new("body", "cat"), TermQuery::new("body", "cat"));
        assert_ne!(TermQuery::new("body", "cat"), TermQuery::new("body", "dog"));
        assert_ne!(TermQuery::new("body", "cat"), TermQuery::new("id", "cat"));
    }

    #[test]
    fn boolean_query_default_is_all_empty_clause_lists() {
        let q = BooleanQuery::new();
        assert!(q.must.is_empty());
        assert!(q.should.is_empty());
        assert!(q.must_not.is_empty());
    }

    #[test]
    fn boolean_query_builder_methods_populate_each_clause_bucket() {
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "bird")]);
        assert_eq!(q.must, vec![TermQuery::new("body", "cat")]);
        assert_eq!(q.should, vec![TermQuery::new("body", "dog")]);
        assert_eq!(q.must_not, vec![TermQuery::new("body", "bird")]);
    }
}
