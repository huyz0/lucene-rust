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

/// `WildcardQuery`-equivalent (`org.apache.lucene.search.WildcardQuery`), task
/// #34's addition: a field plus a glob `pattern` (`*` = zero-or-more bytes, `?`
/// = exactly one Unicode codepoint, `\` escapes the following byte to a plain
/// literal even if it's `*`/`?`/`\`) matched against every term indexed for
/// `field` — the matched set is the **union** of every matching term's
/// postings (see [`crate::resolve_clause_docs`]'s `Clause::Wildcard` arm),
/// mirroring real `WildcardQuery`'s `MultiTermQuery`-style "match any term the
/// automaton accepts" semantics.
///
/// **Why `pattern: Vec<u8>` instead of `String`**: terms in this port are raw
/// `Vec<u8>` (see `TermQuery.term`'s own doc comment) with no guaranteed UTF-8
/// validity, and [`lucene_codecs::wildcard::WildcardPattern`] (the compiled
/// glob this query delegates to — see [`crate::resolve_clause_docs`]) already
/// operates byte-wise. A `String` field would force every caller to already
/// have valid UTF-8 in hand and would need a lossy/fallible conversion back to
/// bytes internally; `Vec<u8>` matches `TermQuery.term`'s own precedent and
/// needs no conversion at match time.
///
/// **Scoring**: unscored/constant, same choice real Lucene's
/// `MultiTermQuery.rewrite()` defaults to for a plain (non-`ConstantScore`-
/// wrapped) multi-term query in modern Lucene — every matching doc scores a
/// flat `1.0` (see [`crate::clause_scores`]'s `Clause::Wildcard` arm), since a
/// wildcard match has no single term's frequency/idf to score against
/// (real Lucene's `MultiTermQuery` documents this default rewrite method as
/// `CONSTANT_SCORE_BLENDED_REWRITE`, which is unscored in exactly this sense —
/// this port doesn't attempt idf-blended constant scoring across the matched
/// terms, just the flat `1.0` a caller can rescale via `Clause::Boost` if it
/// ever needs to, the same way `ConstantScoreQuery`/`BoostQuery` already
/// compose with any other clause).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WildcardQuery {
    pub field: String,
    pub pattern: Vec<u8>,
}

impl WildcardQuery {
    pub fn new(field: impl Into<String>, pattern: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            pattern: pattern.into(),
        }
    }
}

/// `PrefixQuery`-equivalent (`org.apache.lucene.search.PrefixQuery`), task
/// #35's addition: a field plus a literal byte-string `prefix` matched against
/// every term indexed for `field` — a term matches iff its bytes start with
/// `prefix` exactly (no glob syntax at all: unlike [`WildcardQuery`], a `*`/
/// `?`/`\` byte inside `prefix` is just another literal byte to match, never a
/// wildcard metacharacter or an escape). The matched set is the **union** of
/// every matching term's postings (see [`crate::resolve_clause_docs`]'s
/// `Clause::Prefix` arm), same "match any accepted term" contract
/// `WildcardQuery` already has, since real `PrefixQuery` is itself a
/// `MultiTermQuery` with exactly this semantics (`PrefixQuery.compile()`
/// builds an automaton for the same "every term starting with X" language
/// `WildcardQuery`'s trailing-unescaped-`*` also expresses).
///
/// **Design decision: wraps [`lucene_codecs::wildcard::WildcardPattern::prefix`]
/// directly, not `WildcardPattern::new` on an escaped-plus-`*` string.** Real
/// Lucene's `PrefixQuery` is functionally "match every term starting with X",
/// which could be built two ways: (a) a thin wrapper constructing a
/// `WildcardQuery` pattern by literal-escaping `prefix` (backslash-escaping
/// every `*`/`?`/`\` byte) and appending an unescaped trailing `*`, reusing
/// `WildcardPattern::new`'s glob parser unchanged, or (b) a direct prefix
/// match with no glob syntax involved at all. This port takes (b) — and it's
/// not even new code: [`lucene_codecs::wildcard::WildcardPattern::prefix`]
/// already exists (added in task #1 for exactly this purpose, see that
/// module's doc comment) and builds its token list directly from `prefix`'s
/// raw bytes as `Literal` tokens plus one trailing `AnyMany`, **never calling
/// `WildcardPattern::new`'s escape-parsing loop at all**. Option (a) was
/// rejected because it would require this query to re-escape `prefix` byte-by-
/// byte before matching could reuse the parser — fiddly and exactly the kind
/// of edge case the task called out: a prefix like `a*b` must match every term
/// starting with the 3 literal bytes `a`, `*`, `b`, not be reinterpreted as
/// "`a`, then anything, then `b`". Building on `WildcardPattern::prefix`
/// sidesteps that risk entirely rather than mitigating it with careful
/// escaping — there is no escaping step to get wrong, since `prefix`'s bytes
/// never pass through anything that treats `*`/`?`/`\` specially.
///
/// **Why `prefix: Vec<u8>` instead of `String`**: same reasoning as
/// [`WildcardQuery::pattern`]'s own doc comment — terms in this port are raw
/// `Vec<u8>` with no guaranteed UTF-8 validity, and `WildcardPattern::prefix`
/// already operates byte-wise.
///
/// **Scoring**: unscored/constant (flat `1.0` per match), same choice
/// `WildcardQuery` makes and for the same reason — see
/// [`crate::clause_scores`]'s `Clause::Prefix` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrefixQuery {
    pub field: String,
    pub prefix: Vec<u8>,
}

impl PrefixQuery {
    pub fn new(field: impl Into<String>, prefix: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            prefix: prefix.into(),
        }
    }
}

/// `FuzzyQuery`-equivalent (`org.apache.lucene.search.FuzzyQuery`), task
/// #42's addition: a field plus a target `term`, matching every term indexed
/// for `field` whose edit distance to `term` is `<= max_edits`, restricted to
/// terms sharing `term`'s first `prefix_length` bytes exactly (real
/// `FuzzyQuery`'s `prefixLength` — an exact-match requirement, not part of
/// the edit-distance budget). The matched set is the **union** of every
/// matching term's postings (see [`crate::resolve_clause_docs`]'s
/// `Clause::Fuzzy` arm), the same "match any term the automaton/predicate
/// accepts" `MultiTermQuery` contract `WildcardQuery`/`PrefixQuery` already
/// have.
///
/// **Defaults mirror real `FuzzyQuery` exactly**: `max_edits` defaults to
/// `2` (`FuzzyQuery.defaultMaxEdits`, `LevenshteinAutomata.
/// MAXIMUM_SUPPORTED_DISTANCE`), `prefix_length` defaults to `0` (no exact-
/// prefix requirement), and `transpositions` defaults to `true` — real
/// `FuzzyQuery`'s own three-arg-vs-more-arg constructor defaults, meaning an
/// adjacent-character swap counts as **one** edit (Damerau-Levenshtein with
/// transpositions), not two (plain Levenshtein), unless a caller explicitly
/// opts out via [`Self::with_transpositions`]. See
/// [`lucene_codecs::fuzzy::edit_distance`]'s doc comment for exactly which
/// edit-distance variant this is and why, and that module's doc comment for
/// this port's byte-vs-Unicode-codepoint scope decision.
///
/// **Why `term: Vec<u8>` instead of `String`**: same reasoning as
/// [`WildcardQuery::pattern`]'s own doc comment — terms in this port are raw
/// `Vec<u8>` with no guaranteed UTF-8 validity, and
/// [`lucene_codecs::fuzzy::FuzzyMatch`] (the matcher this query delegates
/// to) already operates byte-wise.
///
/// **Scoring**: unscored/constant (flat `1.0` per match), same choice
/// `WildcardQuery`/`PrefixQuery` make and for the same reason — see
/// [`crate::clause_scores`]'s `Clause::Fuzzy` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FuzzyQuery {
    pub field: String,
    pub term: Vec<u8>,
    pub max_edits: u8,
    pub prefix_length: usize,
    pub transpositions: bool,
}

impl FuzzyQuery {
    /// Builds a `FuzzyQuery` with real `FuzzyQuery`'s own defaults:
    /// `max_edits = 2`, `prefix_length = 0`, `transpositions = true`.
    pub fn new(field: impl Into<String>, term: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            term: term.into(),
            max_edits: 2,
            prefix_length: 0,
            transpositions: true,
        }
    }

    /// Builder method setting `max_edits` (see this struct's doc comment for
    /// the default and semantics).
    pub fn with_max_edits(mut self, max_edits: u8) -> Self {
        self.max_edits = max_edits;
        self
    }

    /// Builder method setting `prefix_length` (see this struct's doc comment
    /// for the default and semantics).
    pub fn with_prefix_length(mut self, prefix_length: usize) -> Self {
        self.prefix_length = prefix_length;
        self
    }

    /// Builder method setting `transpositions` (see this struct's doc
    /// comment for the default and semantics — this is the flag that
    /// switches between Damerau-Levenshtein-with-transpositions, `true`, and
    /// plain Levenshtein, `false`).
    pub fn with_transpositions(mut self, transpositions: bool) -> Self {
        self.transpositions = transpositions;
        self
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
    /// A nested `ConstantScoreQuery` (task #33's addition) — matches iff the
    /// wrapped clause matches, but always scores exactly `score`, discarding
    /// whatever the wrapped clause's own score would have been; see
    /// [`ConstantScoreQuery`]'s doc comment.
    ConstantScore(Box<ConstantScoreQuery>),
    /// A nested `BoostQuery` (task #33's addition) — matches iff the wrapped
    /// clause matches, scored as the wrapped clause's own score multiplied by
    /// `boost`; see [`BoostQuery`]'s doc comment.
    Boost(Box<BoostQuery>),
    /// A leaf `WildcardQuery` (task #34's addition) -- matches every doc
    /// containing at least one term (for `query.field`) that
    /// `lucene_codecs::wildcard::WildcardPattern` accepts, unscored (flat
    /// `1.0` per match); see [`WildcardQuery`]'s doc comment.
    Wildcard(WildcardQuery),
    /// A leaf `PrefixQuery` (task #35's addition) -- matches every doc
    /// containing at least one term (for `query.field`) starting with
    /// `query.prefix`'s literal bytes, unscored (flat `1.0` per match); see
    /// [`PrefixQuery`]'s doc comment.
    Prefix(PrefixQuery),
    /// A leaf `FuzzyQuery` (task #42's addition) -- matches every doc
    /// containing at least one term (for `query.field`) within
    /// `query.max_edits` edit distance of `query.term` (restricted to terms
    /// sharing `query.term`'s first `query.prefix_length` bytes exactly),
    /// unscored (flat `1.0` per match); see [`FuzzyQuery`]'s doc comment.
    Fuzzy(FuzzyQuery),
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

impl From<ConstantScoreQuery> for Clause {
    fn from(query: ConstantScoreQuery) -> Self {
        Clause::ConstantScore(Box::new(query))
    }
}

impl From<BoostQuery> for Clause {
    fn from(query: BoostQuery) -> Self {
        Clause::Boost(Box::new(query))
    }
}

impl From<WildcardQuery> for Clause {
    fn from(query: WildcardQuery) -> Self {
        Clause::Wildcard(query)
    }
}

impl From<PrefixQuery> for Clause {
    fn from(query: PrefixQuery) -> Self {
        Clause::Prefix(query)
    }
}

impl From<FuzzyQuery> for Clause {
    fn from(query: FuzzyQuery) -> Self {
        Clause::Fuzzy(query)
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

/// `ConstantScoreQuery`-equivalent (`org.apache.lucene.search.ConstantScoreQuery`):
/// wraps any other [`Clause`] and matches exactly the same docs the inner clause
/// matches, but every matching doc scores exactly `score` — the inner clause's own
/// score (whatever it would have been) is discarded entirely, not folded in.
/// Real `ConstantScoreQuery`'s `ConstantScoreWeight`/`ConstantScoreScorer` always
/// scores `boost` (the query's own boost, `1.0` unless wrapped in a `BoostQuery`,
/// see [`crate::clause_scores`]'s `Clause::ConstantScore` arm) regardless of the
/// inner query's own scoring — this port names the field `score` rather than
/// `boost` since that's the value actually reported per match, matching this
/// struct's single-argument constructor semantics rather than real Lucene's
/// broader `Weight`-level boost propagation this port doesn't otherwise model.
///
/// Nests the same way `Clause::Boolean`/`Clause::DisjunctionMax` already do: the
/// wrapped `inner` clause may itself be any `Clause` variant, including another
/// `ConstantScore`/`Boost`, to arbitrary depth.
#[derive(Debug, Clone, PartialEq)]
pub struct ConstantScoreQuery {
    pub inner: Box<Clause>,
    pub score: f32,
}

impl ConstantScoreQuery {
    /// Builds a `ConstantScoreQuery` wrapping `inner`, always scoring `score`
    /// for any doc `inner` matches. `inner` accepts anything convertible to a
    /// [`Clause`], same builder convenience `BooleanQuery::with_must` etc. use.
    pub fn new(inner: impl Into<Clause>, score: f32) -> Self {
        Self {
            inner: Box::new(inner.into()),
            score,
        }
    }
}

/// `BoostQuery`-equivalent (`org.apache.lucene.search.BoostQuery`): wraps any
/// other [`Clause`] and matches exactly the same docs the inner clause matches,
/// scoring each matching doc as the inner clause's own score multiplied by
/// `boost` — real `BoostQuery.BoostWeight.explain`/`scorer`'s exact behavior
/// (a pure multiplicative rescale of the wrapped query's score, unlike
/// `ConstantScoreQuery`'s discard-and-replace).
///
/// Nests the same way `ConstantScoreQuery` does: `inner` may be any `Clause`
/// variant, including another `Boost`/`ConstantScore`, to arbitrary depth (e.g.
/// `BoostQuery` wrapping a `ConstantScoreQuery` multiplies the constant score by
/// `boost`, matching real Lucene's composition of the two).
#[derive(Debug, Clone, PartialEq)]
pub struct BoostQuery {
    pub inner: Box<Clause>,
    pub boost: f32,
}

impl BoostQuery {
    /// Builds a `BoostQuery` wrapping `inner`, scoring `inner`'s own score
    /// multiplied by `boost` for any doc `inner` matches.
    pub fn new(inner: impl Into<Clause>, boost: f32) -> Self {
        Self {
            inner: Box::new(inner.into()),
            boost,
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

    #[test]
    fn constant_score_query_new_stores_inner_and_score() {
        let q = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 2.0);
        assert_eq!(*q.inner, Clause::Term(TermQuery::new("body", "cat")));
        assert_eq!(q.score, 2.0);
    }

    #[test]
    fn clause_from_constant_score_query_wraps_in_boxed_variant() {
        let inner = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 1.5);
        let clause: Clause = inner.clone().into();
        assert_eq!(clause, Clause::ConstantScore(Box::new(inner)));
    }

    #[test]
    fn boost_query_new_stores_inner_and_boost() {
        let q = BoostQuery::new(TermQuery::new("body", "cat"), 3.0);
        assert_eq!(*q.inner, Clause::Term(TermQuery::new("body", "cat")));
        assert_eq!(q.boost, 3.0);
    }

    #[test]
    fn clause_from_boost_query_wraps_in_boxed_variant() {
        let inner = BoostQuery::new(TermQuery::new("body", "cat"), 2.5);
        let clause: Clause = inner.clone().into();
        assert_eq!(clause, Clause::Boost(Box::new(inner)));
    }

    #[test]
    fn prefix_query_new_stores_field_and_prefix_bytes() {
        let q = PrefixQuery::new("body", "ca");
        assert_eq!(q.field, "body");
        assert_eq!(q.prefix, b"ca");
    }

    #[test]
    fn prefix_query_equality_is_field_and_prefix_based() {
        assert_eq!(
            PrefixQuery::new("body", "ca"),
            PrefixQuery::new("body", "ca")
        );
        assert_ne!(
            PrefixQuery::new("body", "ca"),
            PrefixQuery::new("body", "do")
        );
        assert_ne!(PrefixQuery::new("body", "ca"), PrefixQuery::new("id", "ca"));
    }

    #[test]
    fn clause_from_prefix_query_wraps_in_prefix_variant() {
        let clause: Clause = PrefixQuery::new("body", "ca").into();
        assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "ca")));
    }

    #[test]
    fn with_must_accepts_a_prefix_query_clause() {
        let q = BooleanQuery::new().with_must([PrefixQuery::new("body", "ca")]);
        assert_eq!(q.must, vec![Clause::Prefix(PrefixQuery::new("body", "ca"))]);
    }

    #[test]
    fn fuzzy_query_new_uses_real_fuzzy_querys_defaults() {
        let q = FuzzyQuery::new("body", "cat");
        assert_eq!(q.field, "body");
        assert_eq!(q.term, b"cat");
        assert_eq!(q.max_edits, 2);
        assert_eq!(q.prefix_length, 0);
        assert!(q.transpositions);
    }

    #[test]
    fn fuzzy_query_builder_methods_set_each_field() {
        let q = FuzzyQuery::new("body", "cat")
            .with_max_edits(1)
            .with_prefix_length(2)
            .with_transpositions(false);
        assert_eq!(q.max_edits, 1);
        assert_eq!(q.prefix_length, 2);
        assert!(!q.transpositions);
    }

    #[test]
    fn clause_from_fuzzy_query_wraps_in_fuzzy_variant() {
        let clause: Clause = FuzzyQuery::new("body", "cat").into();
        assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "cat")));
    }

    #[test]
    fn with_must_accepts_a_fuzzy_query_clause() {
        let q = BooleanQuery::new().with_must([FuzzyQuery::new("body", "cat")]);
        assert_eq!(q.must, vec![Clause::Fuzzy(FuzzyQuery::new("body", "cat"))]);
    }

    #[test]
    fn with_must_accepts_constant_score_and_boost_query_clauses() {
        let q = BooleanQuery::new().with_must([
            Clause::from(ConstantScoreQuery::new(TermQuery::new("body", "cat"), 1.0)),
            Clause::from(BoostQuery::new(TermQuery::new("body", "dog"), 2.0)),
        ]);
        assert_eq!(q.must.len(), 2);
    }
}
