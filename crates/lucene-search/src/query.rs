//! `TermQuery`-equivalent (`org.apache.lucene.search.TermQuery`), pared down
//! to this slice's scope: a field name plus a single exact term, no scoring
//! metadata attached (`TermQuery` in real Lucene also carries an optional
//! `TermStates` for cross-segment stats reuse — not needed for a
//! single-segment, no-relevance-scoring first cut, see `lib.rs`'s module
//! doc for the full design rationale).

/// A single exact-term lookup against one field, e.g. `TermQuery::new("body",
/// "cat")` — the Rust analogue of `new TermQuery(new Term("body", "cat"))`.
///
/// Derives `Hash` (in addition to `Eq`, already derived above) so it can be
/// used as a cache key, e.g. by [`crate::query_cache::QueryCache`] -- purely
/// additive, since both fields (`String`, `Vec<u8>`) are already `Hash` and
/// nothing about this type's existing behavior changes.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
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

/// `RegexpQuery`-equivalent (`org.apache.lucene.search.RegexpQuery`), task
/// #43's addition: a field plus a `pattern` string (Lucene-regexp-subset
/// syntax — see [`lucene_codecs::regexp::RegexpPattern`]'s module doc for
/// exactly which operators are supported: literals, `.`, `*`/`+`/`?`,
/// `[...]` classes, `(...)` grouping, `|` alternation; no `{n,m}`, `~`, `&`,
/// no named classes) matched **in full** against every term indexed for
/// `field` — real `RegexpQuery` always matches a term's entire length, never
/// a substring (see that module's "whole-term-match convention" section).
/// The matched set is the **union** of every matching term's postings (see
/// [`crate::resolve_clause_docs`]'s `Clause::Regexp` arm), the same "match
/// any term the automaton/predicate accepts" `MultiTermQuery` contract
/// `WildcardQuery`/`PrefixQuery`/`FuzzyQuery` already have.
///
/// **Why `pattern: String` instead of `Vec<u8>`**: unlike
/// [`WildcardQuery::pattern`]/[`FuzzyQuery::term`] (raw glob/target bytes
/// with no syntax to parse), a regexp pattern is itself a small language
/// that must be parsed before it can match anything, and
/// [`lucene_codecs::regexp::RegexpPattern::new`] can fail on unsupported or
/// malformed syntax (surfaced via [`crate::Error::Regexp`] when this clause
/// is resolved — see [`crate::resolve_clause_docs`]'s `Clause::Regexp` arm)
/// — a `String` keeps the un-parsed pattern text human-readable in error
/// messages and in `Debug`/`PartialEq` output, while the *terms* this
/// pattern is matched against remain the usual raw `Vec<u8>` inside
/// `RegexpPattern::matches` itself.
///
/// **Scoring**: unscored/constant (flat `1.0` per match), same choice
/// `WildcardQuery`/`PrefixQuery`/`FuzzyQuery` make and for the same reason —
/// see [`crate::clause_scores`]'s `Clause::Regexp` arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexpQuery {
    pub field: String,
    pub pattern: String,
}

impl RegexpQuery {
    pub fn new(field: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            field: field.into(),
            pattern: pattern.into(),
        }
    }
}

/// `PointRangeQuery`-equivalent (`org.apache.lucene.search.PointRangeQuery`),
/// task #64's addition: a field plus an inclusive `[min, max]` `i64` range --
/// this port's [`crate::query_parser`] produces this from `field:[min TO
/// max]` syntax. **Parsing-only for now**: nothing in this crate resolves a
/// `PointsRange` clause against an actual segment yet (see
/// [`crate::resolve_clause_docs`]'s doc comment and `docs/parity.md` for the
/// exact deferred scope) -- unlike every other leaf `Clause` variant, there
/// is deliberately no `_doc_ids` resolver function paired with this one yet.
/// The eventual resolver is expected to compose with the already-existing
/// [`crate::points_query::search_points_range`] (this struct's `min`/`max`
/// are exactly what that function's `min_packed`/`max_packed` need once
/// encoded via the field's numeric point encoding, e.g.
/// `lucene_codecs::points`' big-endian-flipped-sign-bit convention for
/// `LongPoint`), not reimplemented here.
///
/// Only `i64`-typed bounds are supported (matching this port's existing
/// `LongPoint`/`search_points_range` numeric convention) -- `String`/date
/// range queries are out of scope for this struct.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PointsRangeQuery {
    pub field: String,
    /// Inclusive lower bound. [`crate::query_parser`]'s `*` (open-ended)
    /// syntax on the low end maps to `i64::MIN`.
    pub min: i64,
    /// Inclusive upper bound. [`crate::query_parser`]'s `*` (open-ended)
    /// syntax on the high end maps to `i64::MAX`.
    pub max: i64,
}

impl PointsRangeQuery {
    pub fn new(field: impl Into<String>, min: i64, max: i64) -> Self {
        Self {
            field: field.into(),
            min,
            max,
        }
    }
}

/// `SpanQuery`-equivalent (`org.apache.lucene.queries.spans.SpanQuery` and its
/// three concrete subclasses this port covers: `SpanTermQuery`,
/// `SpanNearQuery`, `SpanOrQuery` — task #55's addition), a genuinely
/// different query family from [`PhraseQuery`]: instead of only reporting
/// "does this doc match", a span query's result is the actual matching
/// **span ranges** (`[start, end)` position pairs) within a doc, which can
/// then compose (a `SpanNear` of `SpanNear`s, etc). See
/// [`crate::span_matches_in_doc`]'s doc comment for the exact matching
/// algorithm and this type's own scope decision below.
///
/// **Scope decision, stated explicitly (see `docs/parity.md`)**: real
/// Lucene's `Spans` is a lazy iterator API (`nextStartPosition`/`nextDoc`/
/// `advance`, `TwoPhaseIterator` integration, buffered "atNextSpans" state for
/// `SpanNearQuery`'s ordered/unordered merge) — substantial machinery whose
/// full port is out of scope here. This port instead computes span matches
/// **directly against a doc's already-decoded position lists**, the same
/// "compute matches directly against decoded data" shape
/// [`crate::phrase_matches_in_doc`]/[`crate::phrase_matches_in_doc_sloppy`]
/// already use for `PhraseQuery` — an honestly-scoped MVP: does a doc contain
/// a valid span for this query, and what are its matching span ranges,
/// computed eagerly rather than via a lazy iterator. Scoring is likewise flat
/// (`1.0` per matching doc, via [`crate::clause_scores`]'s `Clause::Span`
/// arm), matching this crate's existing `Wildcard`/`Prefix`/`Fuzzy`/`Regexp`
/// precedent — real span-aware scoring (`SpanWeight`/`SpanScorer`) is its own
/// separate, unscoped problem.
///
/// **Variants**:
/// - `SpanTerm { field, term }`: a leaf matching a single term — its spans are
///   exactly that term's `(position, position + 1)` occurrences in a doc
///   (every occurrence, not just "does it occur" — real `SpanTermQuery`'s
///   exact semantics).
/// - `SpanNear { clauses, slop, in_order }`: every sub-`SpanQuery` in
///   `clauses` must have a span within `slop` of each other in the same doc.
///   `in_order == true` requires the sub-spans to appear left-to-right in
///   `clauses`' own order (real `SpanNearQuery(clauses, slop, true)`);
///   `in_order == false` allows the sub-spans in **any** relative order,
///   provided they still fit within a `slop`-sized window (real
///   `SpanNearQuery(clauses, slop, false)`) — this is the capability
///   [`PhraseQuery`]'s own sloppy matching (task #28) deliberately does *not*
///   support (that was explicitly scoped to in-order-only; see
///   [`crate::phrase_matches_in_doc_sloppy`]'s doc comment), making
///   `in_order == false` this type's key differentiator from a sloppy phrase.
/// - `SpanOr { clauses }`: the union of every sub-`SpanQuery`'s own spans —
///   a doc/position matches iff **any** sub-query's spans match there (real
///   `SpanOrQuery`'s exact semantics, the same "pure union" contract
///   [`DisjunctionMaxQuery`] already uses for whole-doc matching, here
///   applied at the span-range granularity instead).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanQuery {
    /// A leaf span matching a single term's occurrences, real
    /// `SpanTermQuery`'s equivalent.
    SpanTerm { field: String, term: Vec<u8> },
    /// `slop` sub-spans of `clauses` required within `slop` total positional
    /// slack of each other; `in_order` selects real `SpanNearQuery`'s
    /// in-order vs any-order semantics — see this enum's doc comment.
    SpanNear {
        clauses: Vec<SpanQuery>,
        slop: u32,
        in_order: bool,
    },
    /// The union of every sub-`SpanQuery`'s own spans, real `SpanOrQuery`'s
    /// equivalent — see this enum's doc comment.
    SpanOr { clauses: Vec<SpanQuery> },
}

impl SpanQuery {
    /// Builds a leaf `SpanTerm` span query for `field`/`term`.
    pub fn span_term(field: impl Into<String>, term: impl Into<Vec<u8>>) -> Self {
        SpanQuery::SpanTerm {
            field: field.into(),
            term: term.into(),
        }
    }

    /// Builds a `SpanNear` span query over `clauses`, requiring every
    /// sub-span within `slop` total positional slack, `in_order` selecting
    /// real `SpanNearQuery`'s in-order vs any-order semantics — see
    /// [`SpanQuery`]'s doc comment.
    pub fn span_near(
        clauses: impl IntoIterator<Item = SpanQuery>,
        slop: u32,
        in_order: bool,
    ) -> Self {
        SpanQuery::SpanNear {
            clauses: clauses.into_iter().collect(),
            slop,
            in_order,
        }
    }

    /// Builds a `SpanOr` span query unioning every sub-`SpanQuery`'s own
    /// spans — see [`SpanQuery`]'s doc comment.
    pub fn span_or(clauses: impl IntoIterator<Item = SpanQuery>) -> Self {
        SpanQuery::SpanOr {
            clauses: clauses.into_iter().collect(),
        }
    }
}

/// `MatchAllDocsQuery`-equivalent (`org.apache.lucene.search.MatchAllDocsQuery`):
/// matches every **live** (non-deleted) doc in a segment, scoring each match a
/// flat `1.0` — real `MatchAllDocsQuery`'s `ConstantScoreScorer`/`ConstantScoreWeight`
/// always score `boost` (the query's own boost, `1.0` unless wrapped in a
/// `BoostQuery`/`Clause::Boost`) regardless of any per-doc statistic, so `1.0`
/// unwrapped is exactly this query's own score, matching this crate's existing
/// `ConstantScoreQuery`/`BoostQuery` composition convention -- a caller wanting a
/// different constant just wraps this in `Clause::ConstantScore`/`Clause::Boost`
/// the same way it already would for any other clause.
///
/// **Why `max_doc: i32` lives on the query itself, not threaded as a new
/// parameter through `resolve_clause_docs`/`clause_scores`/`search_boolean_query`
/// and friends**: every other leaf `Clause` variant resolves its matched-doc set
/// from a term dictionary lookup (a term's own postings list already enumerates
/// exactly the docs it needs), so none of those call sites need to know a
/// segment's `maxDoc` at all. `MatchAllDocsQuery` is the first clause with
/// nothing to seek into -- "every doc" only means something once `maxDoc` is
/// known -- so rather than adding a `max_doc: i32` parameter to every function in
/// `resolve_clause_docs`'s call graph (a wide, purely-mechanical signature change
/// touching every existing call site, including in other crates' tests, for a
/// value only this one variant needs), the caller building the query supplies
/// `max_doc` once, at construction time, exactly the same way it already knows
/// and passes `live_docs` per search call. This mirrors
/// [`crate::doc_value_query::search_numeric_range`]'s own `max_doc: i32`
/// parameter (that function's full `[0, max_doc)` sweep is the same "no
/// dictionary to seek into" shape this query needs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatchAllDocsQuery {
    pub max_doc: i32,
}

impl MatchAllDocsQuery {
    pub fn new(max_doc: i32) -> Self {
        Self { max_doc }
    }
}

/// `MatchNoDocsQuery`-equivalent (`org.apache.lucene.search.MatchNoDocsQuery`):
/// matches nothing, ever, regardless of segment contents or `live_docs` --
/// real `MatchNoDocsQuery.createWeight` returns a `Weight` whose `scorer` is
/// always `null`, the same "no doc ever collected" outcome
/// [`crate::resolve_clause_docs`]'s `Clause::MatchNoDocs` arm returns directly
/// (an empty `Vec`, no segment lookup at all -- there is nothing to look up).
///
/// `reason` mirrors real `MatchNoDocsQuery(String reason)`'s documented
/// human-readable explanation of *why* nothing matches (e.g. what rewrite rule
/// produced this query) -- purely informational, `Default`/`PartialEq`/`Eq`
/// included so it composes with the rest of this module's derive conventions,
/// but nothing in this crate's matching/scoring logic ever inspects it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MatchNoDocsQuery {
    pub reason: String,
}

impl MatchNoDocsQuery {
    /// Builds a `MatchNoDocsQuery` with an empty `reason` (see
    /// [`Self::with_reason`] to set one).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method setting `reason` (see this struct's doc comment).
    pub fn with_reason(mut self, reason: impl Into<String>) -> Self {
        self.reason = reason.into();
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
    /// A leaf `RegexpQuery` (task #43's addition) -- matches every doc
    /// containing at least one term (for `query.field`) that
    /// `lucene_codecs::regexp::RegexpPattern` accepts (matching the term in
    /// full, see that module's whole-term-match convention), unscored (flat
    /// `1.0` per match); see [`RegexpQuery`]'s doc comment.
    Regexp(RegexpQuery),
    /// A `SpanQuery` (task #55's addition, `SpanTerm`/`SpanNear`/`SpanOr`) --
    /// matches every doc with at least one non-empty span (see
    /// [`crate::span_matches_in_doc`]), unscored (flat `1.0` per match, same
    /// convention as `Wildcard`/`Prefix`/`Fuzzy`/`Regexp` above); see
    /// [`SpanQuery`]'s doc comment for the exact span-matching semantics and
    /// this port's scope decision.
    Span(SpanQuery),
    /// A leaf `PointRangeQuery` (task #64's addition) -- parsing-only for
    /// now, see [`PointsRangeQuery`]'s doc comment for the exact deferred
    /// execution scope; no `resolve_clause_docs`/`clause_scores` arm exists
    /// for this variant yet.
    PointsRange(PointsRangeQuery),
    /// A leaf `MatchAllDocsQuery` -- matches every live doc in
    /// `0..query.max_doc`, scored flat `1.0` per match; see
    /// [`MatchAllDocsQuery`]'s doc comment.
    MatchAllDocs(MatchAllDocsQuery),
    /// A leaf `MatchNoDocsQuery` -- matches nothing, ever; see
    /// [`MatchNoDocsQuery`]'s doc comment.
    MatchNoDocs(MatchNoDocsQuery),
}

impl Clause {
    /// Recursively rewrites this clause, applying [`BooleanQuery::rewrite`]'s
    /// simplifications wherever a `Clause::Boolean` occurs, and rewriting the
    /// contents of every other nesting variant (`DisjunctionMax`/
    /// `ConstantScore`/`Boost`) so their children are simplified too --
    /// leaves (`Term`/`Phrase`/`Wildcard`/`Prefix`/`Fuzzy`/`Regexp`/`Span`)
    /// pass through unchanged, since none of them nest a sub-`Clause`.
    ///
    /// See [`BooleanQuery::rewrite`]'s doc comment for the exact rewrite
    /// rules this delegates to for `Clause::Boolean`; `DisjunctionMax`/
    /// `ConstantScore`/`Boost` themselves are never collapsed away (this
    /// port implements no simplification for those three), only their
    /// wrapped clause(s) are rewritten.
    pub fn rewrite(self) -> Clause {
        match self {
            Clause::Boolean(boxed) => (*boxed).rewrite(),
            Clause::DisjunctionMax(boxed) => {
                let DisjunctionMaxQuery {
                    disjuncts,
                    tie_breaker,
                } = *boxed;
                Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery {
                    disjuncts: disjuncts.into_iter().map(Clause::rewrite).collect(),
                    tie_breaker,
                }))
            }
            Clause::ConstantScore(boxed) => {
                let ConstantScoreQuery { inner, score } = *boxed;
                Clause::ConstantScore(Box::new(ConstantScoreQuery {
                    inner: Box::new(inner.rewrite()),
                    score,
                }))
            }
            Clause::Boost(boxed) => {
                let BoostQuery { inner, boost } = *boxed;
                Clause::Boost(Box::new(BoostQuery {
                    inner: Box::new(inner.rewrite()),
                    boost,
                }))
            }
            leaf => leaf,
        }
    }
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

impl From<RegexpQuery> for Clause {
    fn from(query: RegexpQuery) -> Self {
        Clause::Regexp(query)
    }
}

impl From<PointsRangeQuery> for Clause {
    fn from(query: PointsRangeQuery) -> Self {
        Clause::PointsRange(query)
    }
}

impl From<SpanQuery> for Clause {
    fn from(query: SpanQuery) -> Self {
        Clause::Span(query)
    }
}

impl From<MatchAllDocsQuery> for Clause {
    fn from(query: MatchAllDocsQuery) -> Self {
        Clause::MatchAllDocs(query)
    }
}

impl From<MatchNoDocsQuery> for Clause {
    fn from(query: MatchNoDocsQuery) -> Self {
        Clause::MatchNoDocs(query)
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

    /// `BooleanQuery.rewrite()`-equivalent: a pure, semantics-preserving
    /// simplification pass, **opt-in** -- consumes `self` and returns the
    /// simplified [`Clause`] tree, but is never called by
    /// [`crate::search_boolean_query`]/[`crate::search_boolean_query_scored`]
    /// themselves. A caller wanting rewritten queries applies this explicitly
    /// before executing; existing callers that don't call it see zero change
    /// in behavior.
    ///
    /// Rules implemented, precisely (each proven to change neither the
    /// matched-doc set nor per-doc scores -- see
    /// `crates/lucene-search/tests/boolean_query_fixtures.rs`'s
    /// `rewrite_produces_identical_scored_results_*` tests, which run the same
    /// query pre- and post-rewrite against a real fixture segment and assert
    /// identical `search_boolean_query_scored` output):
    ///
    /// 1. **Single-clause unwrap.** A query with exactly one clause total,
    ///    no `must_not`, collapses to that one (already-recursively-rewritten)
    ///    clause directly:
    ///    - `must.len() == 1`, `should` empty, `must_not` empty,
    ///      `minimum_should_match == 0` -- collapses to the sole `must`
    ///      clause. (`minimum_should_match > 0` with an empty `should` list is
    ///      *not* a no-op: [`crate::matched_boolean_docs`] makes that
    ///      combination match nothing, since no doc can ever reach a positive
    ///      threshold against zero `should` clauses -- collapsing here would
    ///      silently turn "matches nothing" into "matches whatever `must`
    ///      matches", so this case is deliberately excluded.)
    ///    - `should.len() == 1`, `must` empty, `must_not` empty,
    ///      `minimum_should_match <= 1` -- collapses to the sole `should`
    ///      clause. (A lone `should` clause with `minimum_should_match` of `0`
    ///      or `1` is already exactly a plain disjunction of one clause, real
    ///      Lucene's own "at least one should must match" floor -- see
    ///      [`crate::matched_boolean_docs`]'s doc comment. A `minimum_should_match`
    ///      greater than `1` is excluded because it can never be satisfied by a
    ///      single clause.)
    ///
    ///    A **pure `must_not`-only** query (or an empty query) is *not*
    ///    collapsed to anything positive: real `BooleanQuery.rewrite()`
    ///    treats both as `MatchNoDocsQuery`
    ///    ([`crate::matched_boolean_docs`]'s doc comment; also task #60's
    ///    confirmed finding), and this port's executor already produces that
    ///    result with no rewrite needed -- see rule 2.
    ///
    /// 2. **Zero clauses / `must_not`-only -> matches nothing.** This port has
    ///    no separate `MatchNoDocsQuery`-equivalent `Clause` variant to
    ///    rewrite *to* -- [`crate::matched_boolean_docs`] already treats
    ///    "`must` and `should` both empty" (including the `must_not`-only and
    ///    the fully-empty case) as "matches nothing" directly, with no
    ///    rewrite required to get that behavior. This rule is therefore a
    ///    **no-op in code** (this function leaves such a `BooleanQuery`
    ///    structurally unchanged, wrapped back in `Clause::Boolean`) --
    ///    documented and tested (see `boolean_query_default_is_all_empty_clause_lists`-
    ///    adjacent rewrite tests below and
    ///    `empty_boolean_query_matches_nothing`/`pure_must_not_query_matches_nothing`
    ///    in `boolean_query_fixtures.rs`) rather than silently assumed.
    ///
    /// 3. **Recursive.** Every clause in `must`/`should`/`must_not` is
    ///    rewritten (via [`Clause::rewrite`]) *before* this function checks
    ///    whether the parent itself simplifies, so a `Clause::Boolean` nested
    ///    arbitrarily deep is simplified bottom-up, and a parent that becomes
    ///    single-clause only after its own child collapsed still collapses
    ///    correctly.
    ///
    /// **Deliberately NOT implemented: duplicate-clause deduplication.** Task
    /// #60 (see `PLAN.md`) already confirmed, against this port's real
    /// executor code (not assumed), that a literal duplicate `should` clause
    /// **double-counts** toward `minimum_should_match` and **double-scores**
    /// in `search_boolean_query_scored` -- real Lucene's own actual
    /// (non-deduping) `BooleanWeight` behavior, not a bug. The same is true
    /// for a duplicate `must` clause: [`crate::clause_scores`] sums every
    /// `must`/`should` clause's own per-doc score, so two identical `must`
    /// clauses contribute that clause's score *twice*. Deduplicating either
    /// would therefore silently change scores (and, for `should` clauses
    /// under `minimum_should_match`, potentially change which docs match) --
    /// the opposite of this function's "matching/scoring never changes"
    /// contract. This rule is skipped rather than implemented incorrectly;
    /// revisit only if a future task adds a scoring model where duplicate
    /// clauses are provably inert.
    pub fn rewrite(self) -> Clause {
        let must: Vec<Clause> = self.must.into_iter().map(Clause::rewrite).collect();
        let should: Vec<Clause> = self.should.into_iter().map(Clause::rewrite).collect();
        let must_not: Vec<Clause> = self.must_not.into_iter().map(Clause::rewrite).collect();
        let minimum_should_match = self.minimum_should_match;

        if must_not.is_empty() && minimum_should_match == 0 && must.len() == 1 && should.is_empty()
        {
            return must.into_iter().next().expect("len checked above");
        }
        if must_not.is_empty() && minimum_should_match <= 1 && should.len() == 1 && must.is_empty()
        {
            return should.into_iter().next().expect("len checked above");
        }

        Clause::Boolean(Box::new(BooleanQuery {
            must,
            should,
            must_not,
            minimum_should_match,
        }))
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
    fn match_all_docs_query_new_stores_max_doc() {
        let q = MatchAllDocsQuery::new(5);
        assert_eq!(q.max_doc, 5);
    }

    #[test]
    fn clause_from_match_all_docs_query_wraps_in_variant() {
        let clause: Clause = MatchAllDocsQuery::new(5).into();
        assert_eq!(clause, Clause::MatchAllDocs(MatchAllDocsQuery::new(5)));
    }

    #[test]
    fn match_no_docs_query_default_has_empty_reason() {
        let q = MatchNoDocsQuery::new();
        assert_eq!(q.reason, "");
    }

    #[test]
    fn match_no_docs_query_with_reason_sets_the_field() {
        let q = MatchNoDocsQuery::new().with_reason("rewrite collapsed to nothing");
        assert_eq!(q.reason, "rewrite collapsed to nothing");
    }

    #[test]
    fn clause_from_match_no_docs_query_wraps_in_variant() {
        let clause: Clause = MatchNoDocsQuery::new().into();
        assert_eq!(clause, Clause::MatchNoDocs(MatchNoDocsQuery::new()));
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
    fn regexp_query_new_stores_field_and_pattern() {
        let q = RegexpQuery::new("body", "ca.*");
        assert_eq!(q.field, "body");
        assert_eq!(q.pattern, "ca.*");
    }

    #[test]
    fn regexp_query_equality_is_field_and_pattern_based() {
        assert_eq!(
            RegexpQuery::new("body", "ca.*"),
            RegexpQuery::new("body", "ca.*")
        );
        assert_ne!(
            RegexpQuery::new("body", "ca.*"),
            RegexpQuery::new("body", "do.*")
        );
        assert_ne!(
            RegexpQuery::new("body", "ca.*"),
            RegexpQuery::new("id", "ca.*")
        );
    }

    #[test]
    fn clause_from_regexp_query_wraps_in_regexp_variant() {
        let clause: Clause = RegexpQuery::new("body", "ca.*").into();
        assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "ca.*")));
    }

    #[test]
    fn with_must_accepts_a_regexp_query_clause() {
        let q = BooleanQuery::new().with_must([RegexpQuery::new("body", "ca.*")]);
        assert_eq!(
            q.must,
            vec![Clause::Regexp(RegexpQuery::new("body", "ca.*"))]
        );
    }

    #[test]
    fn rewrite_collapses_single_must_clause_with_no_should_or_must_not() {
        let q = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn rewrite_collapses_single_should_clause_with_default_minimum_should_match() {
        let q = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
        assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn rewrite_collapses_single_should_clause_with_minimum_should_match_one() {
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat")])
            .with_minimum_should_match(1);
        assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn rewrite_does_not_collapse_single_must_when_minimum_should_match_is_positive() {
        // must=[cat], should=[], minimum_should_match=1 matches nothing (no `should`
        // clause can ever reach the threshold) -- collapsing to a bare `cat` clause
        // would silently turn "matches nothing" into "matches whatever cat matches",
        // so this combination must NOT collapse.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_minimum_should_match(1);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                should: vec![],
                must_not: vec![],
                minimum_should_match: 1,
            }))
        );
    }

    #[test]
    fn rewrite_does_not_collapse_single_should_when_minimum_should_match_exceeds_one() {
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat")])
            .with_minimum_should_match(2);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                should: vec![Clause::Term(TermQuery::new("body", "cat"))],
                must_not: vec![],
                minimum_should_match: 2,
            }))
        );
    }

    #[test]
    fn rewrite_does_not_collapse_single_must_clause_with_a_must_not_present() {
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "dog")]);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                should: vec![],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_does_not_collapse_a_pure_must_not_only_query() {
        // Real BooleanQuery.rewrite() treats a pure must_not query as
        // MatchNoDocsQuery, not as "the must_not clause itself" -- confirm this
        // rewrite leaves it structurally intact (the executor already treats it as
        // matching nothing with no rewrite needed -- see `matched_boolean_docs`).
        let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                should: vec![],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_leaves_an_empty_boolean_query_structurally_unchanged() {
        let q = BooleanQuery::new();
        assert_eq!(
            q.clone().rewrite(),
            Clause::Boolean(Box::new(BooleanQuery::new()))
        );
    }

    #[test]
    fn rewrite_does_not_collapse_when_more_than_one_clause_is_present() {
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![
                    Clause::Term(TermQuery::new("body", "cat")),
                    Clause::Term(TermQuery::new("body", "dog")),
                ],
                should: vec![],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_recurses_into_a_nested_boolean_must_clause_before_checking_the_parent() {
        // inner: must=[cat] alone -> collapses to Term(cat). Outer: must=[inner]
        // alone (after inner's own collapse) -> the *outer* BooleanQuery now also
        // has exactly one must clause and no should/must_not, so it collapses too,
        // all the way down to the bare leaf term.
        let inner = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let outer = BooleanQuery::new().with_must([inner]);
        assert_eq!(outer.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn rewrite_recurses_into_a_nested_boolean_clause_that_does_not_itself_collapse() {
        // inner has two must clauses, so it does NOT collapse -- but it must still
        // come back as a rewritten (structurally-normalized) nested Boolean clause,
        // proving the recursion actually reaches nested clauses rather than only
        // rewriting the top level.
        let inner = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let outer = BooleanQuery::new()
            .with_should([inner.clone()])
            .with_must([TermQuery::new("body", "bird")]);
        let rewritten = outer.rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "bird"))],
                should: vec![Clause::Boolean(Box::new(inner))],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_recurses_into_disjunction_max_disjuncts() {
        let single_must = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let dismax = DisjunctionMaxQuery::new([Clause::from(single_must)], 0.5);
        let clause: Clause = dismax.into();
        assert_eq!(
            clause.rewrite(),
            Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
                [Clause::Term(TermQuery::new("body", "cat"))],
                0.5
            )))
        );
    }

    #[test]
    fn rewrite_recurses_into_constant_score_inner_clause() {
        let single_must = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
        let csq = ConstantScoreQuery::new(single_must, 2.0);
        let clause: Clause = csq.into();
        assert_eq!(
            clause.rewrite(),
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                2.0
            )))
        );
    }

    #[test]
    fn rewrite_recurses_into_boost_inner_clause() {
        let single_should = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
        let bq = BoostQuery::new(single_should, 3.0);
        let clause: Clause = bq.into();
        assert_eq!(
            clause.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                TermQuery::new("body", "cat"),
                3.0
            )))
        );
    }

    #[test]
    fn rewrite_leaves_leaf_clauses_unchanged() {
        assert_eq!(
            Clause::Wildcard(WildcardQuery::new("body", "ca*")).rewrite(),
            Clause::Wildcard(WildcardQuery::new("body", "ca*"))
        );
        assert_eq!(
            Clause::Prefix(PrefixQuery::new("body", "ca")).rewrite(),
            Clause::Prefix(PrefixQuery::new("body", "ca"))
        );
        assert_eq!(
            Clause::Fuzzy(FuzzyQuery::new("body", "cat")).rewrite(),
            Clause::Fuzzy(FuzzyQuery::new("body", "cat"))
        );
        assert_eq!(
            Clause::Regexp(RegexpQuery::new("body", "ca.*")).rewrite(),
            Clause::Regexp(RegexpQuery::new("body", "ca.*"))
        );
        assert_eq!(
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"])).rewrite(),
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
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
