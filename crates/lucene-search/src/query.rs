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

/// One `must`/`should`/`must_not` slot in a [`BooleanQuery`] — a leaf
/// `TermQuery`, a leaf `PhraseQuery` (task #29's addition, closing the gap this
/// enum's doc comment previously flagged), or a nested `BooleanQuery`
/// (recursively, to arbitrary depth: a `Clause::Boolean` can itself contain any
/// of the three variants). The Rust analogue of real `BooleanQuery.add(Query,
/// Occur)` accepting any `Query` implementation into a clause list — this port
/// has exactly three query shapes that need to nest inside a `BooleanQuery`
/// today (a bare term, a phrase, or another boolean combination), so a closed
/// three-variant enum captures the real requirement without speculative
/// generality (see the `rust-performance` skill's "enums where the closed set
/// allows" guidance).
///
/// `Boolean` boxes its nested `BooleanQuery` so `Clause`'s own size doesn't scale
/// with the depth of whatever query tree is embedded inside it — a `BooleanQuery`
/// containing a `Vec<Clause>` would otherwise be an infinitely-sized type without
/// the indirection.
// Only `PartialEq`, not `Eq`: `Clause::DisjunctionMax` embeds a `tie_breaker:
// f32` (task #32), and `f32` has no total order (`NaN`) so it can't derive
// `Eq`. Nothing in this crate needs `Clause: Eq` (no `HashSet<Clause>`/`BTreeSet<Clause>`
// use) -- every existing `assert_eq!`/`==` call site only needs `PartialEq`.
#[derive(Debug, Clone, PartialEq)]
pub enum Clause {
    /// A leaf exact-term clause.
    Term(TermQuery),
    /// A leaf phrase clause — matched via [`crate::search_phrase_query`]'s
    /// matching logic and, for `search_boolean_query_scored`, scored via
    /// [`crate::search_phrase_query_scored`]'s scoring logic (see
    /// [`crate::resolve_clause_docs`]/[`crate::clause_scores`] for exactly how
    /// this wiring works inside a `BooleanQuery`).
    Phrase(PhraseQuery),
    /// A nested `BooleanQuery`, matched (and, for `search_boolean_query_scored`,
    /// scored) against its own `must`/`should`/`must_not`/`minimum_should_match`
    /// independently of the parent query's — see [`crate::search_boolean_query`]'s
    /// doc comment for the exact recursive semantics.
    Boolean(Box<BooleanQuery>),
    /// A nested `DisjunctionMaxQuery` (task #32's addition) — matched (a doc
    /// matches iff any disjunct matches) and scored (real Lucene's `max +
    /// tieBreaker * sum(rest)` dismax formula) via
    /// [`crate::resolve_clause_docs`]/[`crate::clause_scores`], same recursive
    /// treatment as `Clause::Boolean`.
    DisjunctionMax(Box<DisjunctionMaxQuery>),
}

impl From<TermQuery> for Clause {
    fn from(query: TermQuery) -> Self {
        Clause::Term(query)
    }
}

impl From<PhraseQuery> for Clause {
    fn from(query: PhraseQuery) -> Self {
        Clause::Phrase(query)
    }
}

impl From<BooleanQuery> for Clause {
    fn from(query: BooleanQuery) -> Self {
        Clause::Boolean(Box::new(query))
    }
}

impl From<DisjunctionMaxQuery> for Clause {
    fn from(query: DisjunctionMaxQuery) -> Self {
        Clause::DisjunctionMax(Box::new(query))
    }
}

/// `BooleanQuery`-equivalent (`org.apache.lucene.search.BooleanQuery`), pared down to
/// this slice's scope: a flat list of [`Clause`]s (each a `TermQuery`, a
/// `PhraseQuery`, or a nested `BooleanQuery`, recursively — see `Clause`'s doc
/// comment) per `Occur`
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
// Only `PartialEq`, not `Eq` -- see `Clause`'s derive-list note (this struct's
// `Vec<Clause>` fields propagate the same `f32`-via-`DisjunctionMax` reason).
#[derive(Debug, Clone, Default, PartialEq)]
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
/// implicit consecutive term positions: `terms` are always at query-relative
/// positions `0, 1, ..., terms.len() - 1` in phrase order (real
/// `PhraseQuery.Builder.add(Term, int position)` lets a caller attach an arbitrary
/// per-term position for non-adjacent phrase terms — this port has none of that,
/// see the `Vec<Vec<u8>>` note below). `slop` (default `0`, matching real
/// `PhraseQuery.Builder`'s default) is real `PhraseQuery`'s sloppy-matching budget:
/// with `slop == 0` a doc matches iff every term occurs in the field *and* there's
/// some base position `p` such that `terms[i]` occurs at position `p + i` for every
/// `i` (exact adjacency); with `slop > 0`, terms may be spread apart by up to
/// `slop` total positions while staying in phrase order — see
/// [`crate::phrase_matches_in_doc_sloppy`]'s doc comment for the exact formula this
/// port implements (an **in-order-only** subset of real Lucene's sloppy semantics;
/// term reordering within the slop budget is not supported — see that function's
/// doc comment and `docs/parity.md` for the precise scoping).
///
/// **Why `Vec<Vec<u8>>` instead of a `Vec<(Vec<u8>, i32)>` position-annotated list**:
/// with positions always `0..terms.len()`, storing them explicitly would be
/// redundant data a caller could get wrong (e.g. skipping a position) with no
/// non-adjacent-term feature to justify letting them diverge from the implicit
/// sequence — same "don't build the general shape until a second real need shows up"
/// call this crate's `BooleanQuery` doc comment already makes for its clause list.
/// `slop` doesn't change this: it widens how far apart the (still implicitly
/// `0..N`-numbered) terms may drift at match time, it doesn't let a caller assign
/// arbitrary per-term positions.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PhraseQuery {
    pub field: String,
    pub terms: Vec<Vec<u8>>,
    /// Sloppy-matching budget, real `PhraseQuery`'s `slop` parameter. `0` (the
    /// default via [`Self::new`]/`Default`) means exact adjacent matching; see this
    /// struct's doc comment for `slop > 0`'s semantics.
    pub slop: u32,
}

impl PhraseQuery {
    /// Builds an exact (`slop == 0`) phrase query for `terms` in phrase order. An
    /// empty `terms` list is a defined "matches nothing" edge case (mirrors real
    /// `PhraseQuery.Builder.build()`, which returns a `MatchNoDocsQuery` when no terms
    /// were added) — not a panic; see [`crate::search_phrase_query`]'s doc comment.
    /// Use [`Self::with_slop`] to build a sloppy phrase query.
    pub fn new(
        field: impl Into<String>,
        terms: impl IntoIterator<Item = impl Into<Vec<u8>>>,
    ) -> Self {
        Self {
            field: field.into(),
            terms: terms.into_iter().map(Into::into).collect(),
            slop: 0,
        }
    }

    /// Builder method setting `slop` (see this struct's doc comment for exact
    /// semantics), consistent with `BooleanQuery`'s `with_*` builder pattern.
    pub fn with_slop(mut self, slop: u32) -> Self {
        self.slop = slop;
        self
    }
}

/// `DisjunctionMaxQuery`-equivalent (`org.apache.lucene.search.DisjunctionMaxQuery`):
/// a list of `disjuncts` where a doc matches if **any** disjunct matches, scored
/// by real Lucene's `DisjunctionMaxQuery.DisjunctionMaxWeight`/
/// `DisjunctionMaxScorer` formula — the matching disjunct's **maximum** score
/// plus `tie_breaker` times the **sum of every other matching disjunct's
/// score** (see [`crate::clause_scores`]'s `Clause::DisjunctionMax` arm for the
/// exact implementation). `tie_breaker == 0.0` (real
/// `DisjunctionMaxQuery(Collection<Query>)`'s single-arg constructor default)
/// degenerates to pure `max`-of-disjuncts scoring — the same "best matching
/// field wins, others break ties" behavior real Lucene documents for that
/// constructor. Each `disjunct` is a [`Clause`] (any of `Term`/`Phrase`/
/// `Boolean`/`DisjunctionMax`, recursively), same closed-enum nesting pattern
/// `BooleanQuery`'s clause lists already use — see `Clause`'s doc comment.
///
/// **Why `Vec<Clause>` instead of real Lucene's `Collection<Query>`**: this
/// port has exactly four query shapes that need to nest anywhere a `Query` is
/// accepted (`Clause`'s four variants); a `DisjunctionMaxQuery`'s disjuncts are
/// no different from a `BooleanQuery`'s clauses in that respect, so the same
/// closed enum is reused rather than introducing a second, parallel nesting
/// mechanism.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DisjunctionMaxQuery {
    pub disjuncts: Vec<Clause>,
    /// `tieBreakerMultiplier` in real Lucene's constructor. `f32` has no
    /// total order (`NaN`), so this struct — and therefore `Clause`, which
    /// embeds it — derives `PartialEq` only, not `Eq`; see the note on
    /// `Clause`'s own derive list.
    pub tie_breaker: f32,
}

impl DisjunctionMaxQuery {
    pub fn new(disjuncts: impl IntoIterator<Item = impl Into<Clause>>, tie_breaker: f32) -> Self {
        Self {
            disjuncts: disjuncts.into_iter().map(Into::into).collect(),
            tie_breaker,
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
        assert_eq!(q.slop, 0);
    }

    #[test]
    fn phrase_query_default_is_empty() {
        let q = PhraseQuery::default();
        assert_eq!(q.field, "");
        assert!(q.terms.is_empty());
        assert_eq!(q.slop, 0);
    }

    #[test]
    fn phrase_query_with_slop_sets_the_field() {
        let q = PhraseQuery::new("body", ["quick", "fox"]).with_slop(2);
        assert_eq!(q.slop, 2);
    }

    #[test]
    fn clause_from_phrase_query_wraps_in_phrase_variant() {
        let clause: Clause = PhraseQuery::new("body", ["quick", "fox"]).into();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
    }

    #[test]
    fn with_must_accepts_a_phrase_query_clause() {
        let q = BooleanQuery::new().with_must([PhraseQuery::new("body", ["quick", "fox"])]);
        assert_eq!(
            q.must,
            vec![Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))]
        );
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
