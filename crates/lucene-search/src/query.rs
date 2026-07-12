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

/// `PhraseQuery`-equivalent (`org.apache.lucene.search.PhraseQuery`), pared down to
/// **exact adjacent-position matching only (`slop == 0`)**: `terms` are implicitly at
/// consecutive positions `0, 1, ..., terms.len() - 1` in phrase order. Real
/// `PhraseQuery.Builder.add(Term, int position)` lets a caller attach an arbitrary
/// per-term position (for `slop > 0` sloppy matching, or non-adjacent terms) — this
/// port has none of that; a doc matches iff every term occurs in the field *and*
/// there's some base position `p` such that `terms[i]` occurs at position `p + i` for
/// every `i` (see [`crate::search_phrase_query`]'s doc comment for the exact
/// algorithm). Sloppy phrase matching is out of scope for this slice, tracked in
/// `docs/parity.md`.
///
/// **Why `Vec<Vec<u8>>` instead of a `Vec<(Vec<u8>, i32)>` position-annotated list**:
/// with positions always `0..terms.len()`, storing them explicitly would be
/// redundant data a caller could get wrong (e.g. skipping a position) with no
/// slop/non-adjacent-term feature to justify letting them diverge from the implicit
/// sequence — same "don't build the general shape until a second real need shows up"
/// call this crate's `BooleanQuery` doc comment already makes for its clause list.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhraseQuery {
    pub field: String,
    pub terms: Vec<Vec<u8>>,
}

impl PhraseQuery {
    /// Builds a phrase query for `terms` in phrase order. An empty `terms` list is a
    /// defined "matches nothing" edge case (mirrors real
    /// `PhraseQuery.Builder.build()`, which returns a `MatchNoDocsQuery` when no terms
    /// were added) — not a panic; see [`crate::search_phrase_query`]'s doc comment.
    pub fn new(
        field: impl Into<String>,
        terms: impl IntoIterator<Item = impl Into<Vec<u8>>>,
    ) -> Self {
        Self {
            field: field.into(),
            terms: terms.into_iter().map(Into::into).collect(),
        }
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

    #[test]
    fn phrase_query_new_stores_field_and_terms_in_order() {
        let q = PhraseQuery::new("body", ["quick", "brown", "fox"]);
        assert_eq!(q.field, "body");
        assert_eq!(
            q.terms,
            vec![b"quick".to_vec(), b"brown".to_vec(), b"fox".to_vec()]
        );
    }

    #[test]
    fn phrase_query_default_is_empty() {
        let q = PhraseQuery::default();
        assert_eq!(q.field, "");
        assert!(q.terms.is_empty());
    }

    #[test]
    fn phrase_query_equality_is_field_and_terms_based() {
        assert_eq!(
            PhraseQuery::new("body", ["a", "b"]),
            PhraseQuery::new("body", ["a", "b"])
        );
        assert_ne!(
            PhraseQuery::new("body", ["a", "b"]),
            PhraseQuery::new("body", ["a", "c"])
        );
        assert_ne!(
            PhraseQuery::new("body", ["a", "b"]),
            PhraseQuery::new("id", ["a", "b"])
        );
    }
}
