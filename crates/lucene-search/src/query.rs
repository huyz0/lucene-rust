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

/// One `must`/`should`/`must_not` slot in a [`BooleanQuery`] — either a leaf
/// `TermQuery`, or a nested `BooleanQuery` (recursively, to arbitrary depth: a
/// `Clause::Boolean` can itself contain `Clause::Boolean` clauses). The Rust
/// analogue of real `BooleanQuery.add(Query, Occur)` accepting any `Query`
/// implementation into a clause list — this port has exactly two query shapes that
/// need to nest inside a `BooleanQuery` today (a bare term, or another boolean
/// combination of terms), so a closed two-variant enum captures the real
/// requirement without speculative generality (see the `rust-performance` skill's
/// "enums where the closed set allows" guidance, and this module's own
/// `PhraseQuery` doc comment for the same "don't build the general shape until a
/// second real need shows up" call). `PhraseQuery` is deliberately **not** a
/// `Clause` variant yet — phrase queries as boolean clauses are a documented
/// future extension (`docs/parity.md`), not a current need.
///
/// `Boolean` boxes its nested `BooleanQuery` so `Clause`'s own size doesn't scale
/// with the depth of whatever query tree is embedded inside it — a `BooleanQuery`
/// containing a `Vec<Clause>` would otherwise be an infinitely-sized type without
/// the indirection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Clause {
    /// A leaf exact-term clause.
    Term(TermQuery),
    /// A nested `BooleanQuery`, matched (and, for `search_boolean_query_scored`,
    /// scored) against its own `must`/`should`/`must_not`/`minimum_should_match`
    /// independently of the parent query's — see [`crate::search_boolean_query`]'s
    /// doc comment for the exact recursive semantics.
    Boolean(Box<BooleanQuery>),
}

impl From<TermQuery> for Clause {
    fn from(query: TermQuery) -> Self {
        Clause::Term(query)
    }
}

impl From<BooleanQuery> for Clause {
    fn from(query: BooleanQuery) -> Self {
        Clause::Boolean(Box::new(query))
    }
}

/// `BooleanQuery`-equivalent (`org.apache.lucene.search.BooleanQuery`), pared down to
/// this slice's scope: a flat list of [`Clause`]s (each either a `TermQuery` or a
/// nested `BooleanQuery`, recursively — see `Clause`'s doc comment) per `Occur`
/// bucket (`MUST`, `SHOULD`, `MUST_NOT`) plus `minimumNumberShouldMatch` — no
/// `FILTER` (a `FILTER` clause only differs from `MUST` by not contributing to
/// scoring, and this slice has no separate `FILTER` concept yet, so it would be a
/// distinction without a difference here).
///
/// **Why three flat `Vec<Clause>` fields instead of real Lucene's single
/// `Vec<(Occur, Query)>` clause list**: real `BooleanQuery` stores clauses in
/// insertion order because `Occur` is per-clause and clause order matters for some
/// scoring/explain paths. This port has no `explain()` and no clause-order-sensitive
/// scoring, so grouping by `Occur` up front removes a dispatch step
/// `search_boolean_query` would otherwise redo on every call (partition-by-`Occur`),
/// with no loss of information this slice actually uses. If clause order or a
/// separate `FILTER` occur land later, revisit — the `Vec<(Occur, Query)>` shape
/// earns its keep once clause order or a fourth `Occur` matters.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BooleanQuery {
    /// `Occur.MUST`: every doc must match every clause here (conjunction).
    pub must: Vec<Clause>,
    /// `Occur.SHOULD`: interaction with `minimum_should_match` mirrors real
    /// `BooleanQuery`/`BooleanWeight` exactly (verified against
    /// `BooleanWeight.scorer`/`bulkScorer`/`explain`, not guessed — `should` clauses
    /// are gated by `minimum_should_match` **regardless of whether `must` is also
    /// non-empty**; it is not a "should only matters when must is absent" rule).
    /// With `minimum_should_match == 0` (the default): when `must` is non-empty,
    /// `should` is purely score-contributing and does not narrow the matched set;
    /// when `must` is empty, `should`'s disjunction *is* the matched set (a doc
    /// needs at least one `should` hit, which is `minimum_should_match`'s implicit
    /// floor of 1 in that case). With `minimum_should_match > 0`: a doc — whether or
    /// not it already satisfies every `must` clause — must additionally match at
    /// least `minimum_should_match` of the `should` clauses to match at all; see
    /// `search_boolean_query`'s doc comment in `lib.rs` for the exact algorithm.
    pub should: Vec<Clause>,
    /// `Occur.MUST_NOT`: a doc must match none of these clauses.
    pub must_not: Vec<Clause>,
    /// `minimumNumberShouldMatch`-equivalent: the minimum number of `should` clauses
    /// a doc must match, on top of satisfying every `must` clause (if any). `0`
    /// (the default, via `Default`/`new`) means "no minimum" — real `BooleanQuery`'s
    /// own default. Real `BooleanQuery.rewrite()` turns a `should.len() <
    /// minimum_should_match` query into `MatchNoDocsQuery`; this port doesn't
    /// special-case that (see `search_boolean_query`'s doc comment) because the
    /// counting mechanism already yields "no doc can ever reach the threshold" in
    /// that case, the same observable result, with no separate branch needed.
    pub minimum_should_match: usize,
}

impl BooleanQuery {
    pub fn new() -> Self {
        Self::default()
    }

    /// Accepts anything convertible to a [`Clause`] — a bare `TermQuery` (via
    /// `Clause`'s `From<TermQuery>` impl) or an already-built nested `BooleanQuery`
    /// (via `From<BooleanQuery>`), so existing `with_must([TermQuery::new(...)])`
    /// call sites keep compiling unchanged while `with_must([nested_query])` now
    /// also works for a `BooleanQuery` clause.
    pub fn with_must(mut self, clauses: impl IntoIterator<Item = impl Into<Clause>>) -> Self {
        self.must.extend(clauses.into_iter().map(Into::into));
        self
    }

    /// See [`Self::with_must`]'s doc comment for the accepted clause shapes.
    pub fn with_should(mut self, clauses: impl IntoIterator<Item = impl Into<Clause>>) -> Self {
        self.should.extend(clauses.into_iter().map(Into::into));
        self
    }

    /// See [`Self::with_must`]'s doc comment for the accepted clause shapes.
    pub fn with_must_not(mut self, clauses: impl IntoIterator<Item = impl Into<Clause>>) -> Self {
        self.must_not.extend(clauses.into_iter().map(Into::into));
        self
    }

    /// Sets `minimum_should_match` (see the field doc comment for exact semantics).
    /// Builder-style, consistent with `with_must`/`with_should`/`with_must_not`.
    pub fn with_minimum_should_match(mut self, minimum_should_match: usize) -> Self {
        self.minimum_should_match = minimum_should_match;
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
        assert_eq!(q.minimum_should_match, 0);
    }

    #[test]
    fn boolean_query_builder_methods_populate_each_clause_bucket() {
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "bird")]);
        assert_eq!(q.must, vec![Clause::Term(TermQuery::new("body", "cat"))]);
        assert_eq!(q.should, vec![Clause::Term(TermQuery::new("body", "dog"))]);
        assert_eq!(
            q.must_not,
            vec![Clause::Term(TermQuery::new("body", "bird"))]
        );
        assert_eq!(q.minimum_should_match, 0);
    }

    #[test]
    fn clause_from_term_query_wraps_in_term_variant() {
        let clause: Clause = TermQuery::new("body", "cat").into();
        assert_eq!(clause, Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn clause_from_boolean_query_wraps_in_boxed_boolean_variant() {
        let nested = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let clause: Clause = nested.clone().into();
        assert_eq!(clause, Clause::Boolean(Box::new(nested)));
    }

    #[test]
    fn with_must_accepts_a_nested_boolean_query_clause() {
        let nested = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let q = BooleanQuery::new().with_must([nested.clone()]);
        assert_eq!(q.must, vec![Clause::Boolean(Box::new(nested))]);
    }

    #[test]
    fn nested_boolean_clauses_can_recurse_to_multiple_levels() {
        // A 3-level tree: top.must = [inner], inner.must = [innermost], innermost.must
        // = [TermQuery] -- confirms `Clause::Boolean` genuinely nests, not just one
        // extra level.
        let innermost = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let inner = BooleanQuery::new().with_must([innermost.clone()]);
        let top = BooleanQuery::new().with_must([inner.clone()]);

        let Clause::Boolean(top_inner) = &top.must[0] else {
            panic!("expected a nested Boolean clause");
        };
        assert_eq!(**top_inner, inner);
        let Clause::Boolean(inner_innermost) = &top_inner.must[0] else {
            panic!("expected a nested Boolean clause");
        };
        assert_eq!(**inner_innermost, innermost);
    }

    #[test]
    fn boolean_query_with_minimum_should_match_sets_the_field() {
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_minimum_should_match(2);
        assert_eq!(q.minimum_should_match, 2);
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
