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
/// **Scoring**: scored, unlike `WildcardQuery`/`PrefixQuery`/`RegexpQuery`.
/// Real `FuzzyQuery`'s default rewrite method is
/// `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`, not a constant-score
/// one: each expanded term carries a `FuzzyTermsEnum` boost derived from its
/// edit distance, and the terms are scored against one *blended* document
/// frequency. See `crate::fuzzy_doc_scores`.
///
/// **`max_expansions` (task #221)**: real `FuzzyQuery`'s `maxExpansions`
/// (`FuzzyQuery.defaultMaxExpansions = 50`) bounds how many matching terms
/// from the field's term dictionary get expanded into the query's actual
/// execution -- without this cap, a fuzzy query against a huge or adversarial
/// term dictionary could match an unbounded number of terms (a real
/// behavioral difference from Lucene, and a performance/DoS-shaped gap for a
/// query whose target term can originate from untrusted input, the same
/// class of concern task #198's regexp step-budget fix addressed). This
/// port's [`crate::fuzzy_doc_ids`] enforces the cap by `take`ing only the
/// first `max_expansions` terms off
/// [`lucene_codecs::blocktree::FieldTerms::fuzzy_intersect`]'s iterator.
/// Note that a segment's whole term dictionary is already decoded into an
/// in-memory sorted `Vec` at open time (see `BlockTreeFields`'s own doc
/// comment), so `take` doesn't skip any decode/IO work -- it does avoid
/// running the fuzzy-match predicate and allocating a result for every
/// entry in the (already prefix-narrowed) range past the cap.
///
/// **Selection policy when more than `max_expansions` terms match**: real
/// Lucene's. `TopTermsRewrite.collect` keeps a size-`maxExpansions` priority
/// queue ordered by the `FuzzyTermsEnum` boost, dropping the lexicographically
/// later term on a tie, and `crate::fuzzy_expanded_terms` reproduces that
/// ordering. (This port used to keep the first `max_expansions` terms in
/// sorted term-dictionary order instead, which is close to the *opposite*
/// selection: term order is uncorrelated with edit distance.)
///
/// **`max_edits` is not validated here.** Real `FuzzyQuery`'s constructor
/// throws when `maxEdits > LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE`
/// (2), because Lucene ships parametric automaton descriptions only for
/// distances 1 and 2. This port's matcher is a DP with no such ceiling, so a
/// larger value works and simply has no Lucene equivalent -- see
/// [`lucene_codecs::fuzzy::MAXIMUM_SUPPORTED_DISTANCE`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FuzzyQuery {
    pub field: String,
    pub term: Vec<u8>,
    pub max_edits: u8,
    pub prefix_length: usize,
    pub transpositions: bool,
    pub max_expansions: usize,
}

impl FuzzyQuery {
    /// Real `FuzzyQuery.defaultMaxExpansions`: the default cap on how many
    /// matching terms get expanded into the query's execution.
    pub const DEFAULT_MAX_EXPANSIONS: usize = 50;

    /// Builds a `FuzzyQuery` with real `FuzzyQuery`'s own defaults:
    /// `max_edits = 2`, `prefix_length = 0`, `transpositions = true`,
    /// `max_expansions = 50` ([`Self::DEFAULT_MAX_EXPANSIONS`]).
    pub fn new(field: impl Into<String>, term: impl Into<Vec<u8>>) -> Self {
        Self {
            field: field.into(),
            term: term.into(),
            max_edits: 2,
            prefix_length: 0,
            transpositions: true,
            max_expansions: Self::DEFAULT_MAX_EXPANSIONS,
        }
    }

    /// Builder method setting `max_expansions` (see this struct's doc comment
    /// for the default, the early-termination mechanism, and the
    /// term-dictionary-order selection policy used when more terms match
    /// than this cap allows).
    pub fn with_max_expansions(mut self, max_expansions: usize) -> Self {
        self.max_expansions = max_expansions;
        self
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
/// `[...]` classes, `(...)` grouping, `|` alternation, `{n}`/`{n,}`/`{n,m}`
/// bounded repetition; no `~`, `&`, no named classes) matched **in full**
/// against every term indexed for
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
/// [`crate::phrase_matches_in_doc`]/[`crate::sloppy_phrase`]
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
///   `SpanNearQuery(clauses, slop, false)`). Note that a sloppy
///   [`PhraseQuery`] is **not** the same thing as `in_order == false`, even
///   though both admit reordering: a phrase's budget is the width of the
///   window covering every term's *slot-shifted* position
///   (see [`crate::sloppy_phrase`]), while `SpanNearQuery`'s is the summed
///   slack between adjacent spans. `in_order == true` is the arm with no
///   phrase analogue at all.
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

/// `TermInSetQuery`-equivalent (`org.apache.lucene.search.TermInSetQuery`): a
/// field plus a set of exact `terms`, matching every doc containing **any**
/// of them -- semantically a disjunction over same-field `TermQuery`s (real
/// `TermInSetQuery`'s own class doc comment: "by default, behaves like a
/// `ConstantScoreQuery` over a `BooleanQuery` containing only `SHOULD`
/// clauses"), implemented more efficiently than building that full
/// `BooleanQuery` scorer tree by seeking each term directly (see
/// [`crate::resolve_clause_docs`]'s `Clause::TermInSet` arm, which unions
/// each term's own postings the same way [`WildcardQuery`]/[`PrefixQuery`]
/// already union their matched terms' postings).
///
/// **Scoring: unscored/constant, NOT max-of-matched-terms, NOT
/// summed-like-a-`BooleanQuery`-OR.** Verified directly against real
/// `TermInSetQuery.java`'s class doc comment, which states outright: "NOTE:
/// This query produces scores that are equal to its boost" -- i.e. flat
/// `1.0` per matching doc (no boost wrapper) regardless of which or how many
/// of the given terms matched, exactly the same convention
/// `WildcardQuery`/`PrefixQuery`/`FuzzyQuery`/`RegexpQuery` already use in
/// this port (see [`crate::clause_scores`]'s `Clause::TermInSet` arm). This
/// is a real semantic difference from a `BooleanQuery` of `SHOULD` clauses,
/// which sums each matched clause's own idf-based score -- `TermInSetQuery`
/// only matches "like" that shape, it does not score like it (real Lucene's
/// default `CONSTANT_SCORE_BLENDED_REWRITE` rewrite is exactly this: a
/// constant-scoring rewrite, not `SCORING_BOOLEAN_REWRITE`).
///
/// **Why `terms: Vec<Vec<u8>>` instead of a `PrefixCodedTerms`-style packed
/// encoding**: real `TermInSetQuery` sorts, dedupes, and prefix-compresses
/// its terms into a `PrefixCodedTerms` for compact storage and to enable its
/// `SetEnum`'s ping-pong intersection against the terms dict. This port's
/// existing multi-term clauses (`Wildcard`/`Prefix`/`Fuzzy`/`Regexp`) all
/// resolve by intersecting the terms dict once per query and unioning
/// postings, so there's no equivalent packed-storage need here; a plain
/// `Vec<Vec<u8>>` matches [`TermQuery::term`]'s own raw-bytes convention and
/// needs no packing/sorting step. Duplicate terms in `terms` are harmless --
/// [`crate::resolve_clause_docs`]'s `Clause::TermInSet` arm already
/// deduplicates the resulting doc-ID list (same as `WildcardQuery`'s
/// multi-term union), matching real `TermInSetQuery.packTerms`'s own
/// dedup step, just applied to doc IDs downstream rather than to `terms`
/// upfront.
///
/// **Empty `terms`**: matches nothing (an empty union is empty), the same
/// "no doc ever collected" outcome [`MatchNoDocsQuery`] documents, requiring
/// no special-case branch since [`crate::resolve_clause_docs`]'s
/// `Clause::TermInSet` arm's loop over zero terms already produces an empty
/// `Vec` naturally.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TermInSetQuery {
    pub field: String,
    pub terms: Vec<Vec<u8>>,
}

impl TermInSetQuery {
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
    /// A leaf `TermInSetQuery` -- matches every doc containing at least one
    /// of `query.terms` (for `query.field`), unscored (flat `1.0` per
    /// match); see [`TermInSetQuery`]'s doc comment.
    TermInSet(TermInSetQuery),
    /// A leaf `MultiPhraseQuery` -- a phrase whose every position accepts a
    /// *set* of alternative terms; see [`MultiPhraseQuery`]'s doc comment.
    MultiPhrase(MultiPhraseQuery),
}

impl Clause {
    /// Recursively rewrites this clause, applying [`BooleanQuery::rewrite`]'s
    /// simplifications wherever a `Clause::Boolean` occurs, and rewriting the
    /// contents of every other nesting variant (`DisjunctionMax`/
    /// `ConstantScore`/`Boost`) so their children are simplified too --
    /// leaves (`Term`/`Phrase`/`Wildcard`/`Prefix`/`Fuzzy`/`Regexp`/`Span`/
    /// `TermInSet`) pass through unchanged, since none of them nest a
    /// sub-`Clause`.
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

impl From<MultiPhraseQuery> for Clause {
    fn from(query: MultiPhraseQuery) -> Self {
        Clause::MultiPhrase(query)
    }
}

impl From<MatchNoDocsQuery> for Clause {
    fn from(query: MatchNoDocsQuery) -> Self {
        Clause::MatchNoDocs(query)
    }
}

impl From<TermInSetQuery> for Clause {
    fn from(query: TermInSetQuery) -> Self {
        Clause::TermInSet(query)
    }
}

/// `BooleanQuery`-equivalent (`org.apache.lucene.search.BooleanQuery`): a flat list
/// of [`Clause`]s (each a `TermQuery`, a `PhraseQuery`, or a nested
/// `BooleanQuery`, recursively — see `Clause`'s doc comment) per `Occur` bucket,
/// plus `minimumNumberShouldMatch`. All four of Java's
/// [`BooleanClause.Occur`](https://lucene.apache.org/core/10_5_0/core/org/apache/lucene/search/BooleanClause.Occur.html)
/// values are represented: `MUST` ([`Self::must`]), `FILTER` ([`Self::filter`]),
/// `SHOULD` ([`Self::should`]) and `MUST_NOT` ([`Self::must_not`]).
///
/// **Why four flat `Vec<Clause>` fields instead of real Lucene's single
/// `Vec<(Occur, Query)>` clause list**: real `BooleanQuery` stores clauses in
/// insertion order because `Occur` is per-clause, but it *also* immediately
/// partitions them into a `Map<Occur, Collection<Query>>` (`clauseSets`) that
/// every rewrite rule, `BooleanWeight.count` and `BooleanScorerSupplier` then
/// read from. Grouping by `Occur` up front is that map, materialized once,
/// removing a partition step the executor would otherwise redo on every call.
/// The information lost is clause *interleaving* across buckets, which only
/// `BooleanQuery.toString` and `BooleanWeight.explain`'s sub-order depend on;
/// both are emitted here in Java's own `Occur` declaration order
/// (`MUST`, `FILTER`, `SHOULD`, `MUST_NOT`) rather than insertion order.
// Only `PartialEq`, not `Eq` -- see `Clause`'s derive-list note (this struct's
// `Vec<Clause>` fields propagate the same `f32`-via-`DisjunctionMax` reason).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BooleanQuery {
    /// `Occur.MUST`: every doc must match every clause here (conjunction), and
    /// every clause here contributes to the score.
    pub must: Vec<Clause>,
    /// `Occur.FILTER`: "like `MUST` except that these clauses do not participate
    /// in scoring" (`BooleanClause.Occur.FILTER`'s own javadoc). A doc must match
    /// every clause here, exactly as for [`Self::must`], but a filter clause
    /// contributes **zero** to the score and is never summed — real Lucene builds
    /// its `Weight` with `ScoreMode.COMPLETE_NO_SCORES`, and
    /// `BooleanScorerSupplier.req` puts it in `required` but not in
    /// `requiredScoring`, so `ConjunctionScorer.score()` never calls it.
    ///
    /// Two consequences worth stating because they are easy to get wrong:
    ///
    /// - **A filter clause does not count toward [`Self::minimum_should_match`]**
    ///   (nor does a `must_not` one) — `BooleanWeight.explain` increments
    ///   `shouldMatchCount` only for `Occur.SHOULD`.
    /// - **A query of only filter clauses matches, with score `0`.** It is not a
    ///   "pure negative" query: `BooleanQuery.rewrite`'s pure-negative test is
    ///   `clauses.size() == clauseSets.get(MUST_NOT).size()`, which a filter
    ///   clause fails, and its single-clause optimization rewrites a lone filter
    ///   to `BoostQuery(ConstantScoreQuery(q), 0)` — a match at score `0`, not a
    ///   non-match.
    pub filter: Vec<Clause>,
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

/// Order-preserving exact-duplicate removal, keeping the first occurrence --
/// the `HashSet` semantics `BooleanQuery`'s `clauseSets` gives `FILTER` and
/// `MUST_NOT` clauses. A `Vec` scan rather than a hash set because [`Clause`]
/// is only `PartialEq` (it carries `f32` tie-breakers, see `Clause`'s derive
/// note) and clause lists are short.
fn dedup_clauses(clauses: &mut Vec<Clause>) {
    let mut seen: Vec<Clause> = Vec::with_capacity(clauses.len());
    clauses.retain(|clause| {
        if seen.contains(clause) {
            false
        } else {
            seen.push(clause.clone());
            true
        }
    });
}

/// Java's `Deduplicate <occur> clauses by summing up their boosts` block, for
/// one bucket: unwrap each clause's `BoostQuery` chain multiplying the boosts,
/// sum per distinct unwrapped clause, and rebuild **only** when that collapsed
/// the count (Java's `if (map.size() != clauseSets.get(occur).size())`).
/// Returns whether it rebuilt.
///
/// The boost arithmetic is Java's: the product and the sum are `double`
/// (`double boost = 1; boost *= bq.getBoost();` then
/// `getOrDefault(query, 0d) + boost`), narrowed to `f32` once at the end via
/// `entry.getValue().floatValue()`. Summing in `f32` instead would round
/// differently for three or more duplicates, which is a score difference, not
/// a style one.
///
/// A `Vec` scan rather than a hash map, for the same reason
/// [`dedup_clauses`] uses one: [`Clause`] is only `PartialEq` (it carries
/// `f32` tie-breakers) and clause lists are bounded by `maxClauseCount`.
fn dedup_by_boost_sum(clauses: &mut Vec<Clause>) -> bool {
    let mut groups: Vec<(Clause, f64)> = Vec::with_capacity(clauses.len());
    for clause in clauses.iter() {
        let (inner, boost) = unwrap_boost_chain(clause);
        match groups.iter_mut().find(|(q, _)| *q == inner) {
            Some((_, summed)) => *summed += boost,
            None => groups.push((inner, boost)),
        }
    }
    if groups.len() == clauses.len() {
        return false;
    }
    *clauses = groups
        .into_iter()
        .map(|(query, boost)| {
            let boost = boost as f32;
            if boost != 1.0 {
                Clause::Boost(Box::new(BoostQuery::new(query, boost)))
            } else {
                query
            }
        })
        .collect();
    true
}

/// `while (query instanceof BoostQuery) { boost *= bq.getBoost(); query =
/// bq.getQuery(); }` -- the innermost non-`BoostQuery` clause and the product
/// of every boost wrapping it.
fn unwrap_boost_chain(clause: &Clause) -> (Clause, f64) {
    let mut boost = 1.0f64;
    let mut current = clause;
    while let Clause::Boost(b) = current {
        boost *= b.boost as f64;
        current = &b.inner;
    }
    (current.clone(), boost)
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
    /// **What "semantics-preserving" does and does not claim here.** Every rule
    /// below preserves the *matched-doc set* exactly. Per-doc **scores** are
    /// preserved bit-for-bit by rules 1-5 and 8, which is what
    /// `crates/lucene-search/tests/boolean_query_fixtures.rs`'s
    /// `rewrite_produces_identical_scored_results_*` tests assert (same query,
    /// pre- and post-rewrite, against a real fixture segment). Rules 6, 7 and 9
    /// **reorder the clause list [`crate::clause_scores`] folds over**, and
    /// `f32` addition is not associative, so they can move the last bit of a
    /// score. `rewrite_flattening_and_inlining_preserve_scores_bit_for_bit`
    /// (same file) pins the fixture's actual behaviour rather than asserting a
    /// property that does not hold in general; a caller that needs
    /// bit-stability across a rewrite should not rewrite. Java has exactly the
    /// same property and the same reason -- `BooleanScorer` sums in clause
    /// order too.
    ///
    /// Rules implemented, precisely:
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
    ///    A **pure `must_not`-only** query (or an empty query) is not
    ///    collapsed to anything positive -- it collapses to
    ///    `MatchNoDocsQuery`; see rule 2.
    ///
    /// 2. **Zero clauses / `must_not`-only -> `MatchNoDocsQuery`.** Java's
    ///    first two rules, with its two distinct reason strings:
    ///    `MatchNoDocsQuery("empty BooleanQuery")` when there were no clauses
    ///    at all, and `MatchNoDocsQuery("pure negative BooleanQuery")` when
    ///    every clause was `MUST_NOT`.
    ///
    ///    This used to be a **no-op in code**, on the grounds that the port had
    ///    "no separate `MatchNoDocsQuery`-equivalent `Clause` variant to
    ///    rewrite *to*" and that [`crate::matched_boolean_docs`] already
    ///    treated "`must` and `should` both empty" as matching nothing. The
    ///    first half of that stopped being true when [`Clause::MatchNoDocs`]
    ///    was added, and the b12 sweep's rules 5, 8 and 9 all `return
    ///    Clause::MatchNoDocs(..)` -- so the justification no longer held and
    ///    the rule is now implemented. The observable matching behaviour is
    ///    unchanged (both forms match nothing); what changes is that
    ///    `rewrite()` now *reports* it, with Java's reason string, instead of
    ///    handing back a query that looks satisfiable.
    ///
    ///    Java tests this before recursing and catches the post-recursion case
    ///    on `IndexSearcher.rewrite`'s next pass, since that loops to a
    ///    fixpoint. This port's single bottom-up pass tests it after rule 5's
    ///    `MatchNoDocsQuery` drops instead, which reaches the same fixpoint in
    ///    one go: a query whose only `SHOULD` clause rewrote to
    ///    `MatchNoDocsQuery` *is* a pure negative query once that clause is
    ///    gone.
    ///
    /// 3. **Recursive.** Every clause in `must`/`should`/`must_not` is
    ///    rewritten (via [`Clause::rewrite`]) *before* this function checks
    ///    whether the parent itself simplifies, so a `Clause::Boolean` nested
    ///    arbitrarily deep is simplified bottom-up, and a parent that becomes
    ///    single-clause only after its own child collapsed still collapses
    ///    correctly.
    ///
    /// 4. **`must_not` duplicate removal.** Exact duplicate `must_not`
    ///    clauses collapse to one. This mirrors real `BooleanQuery`'s own
    ///    "remove duplicate FILTER and MUST_NOT clauses" rewrite rule (see
    ///    `BooleanQuery.java`'s `rewrite()`, the block starting `// remove
    ///    duplicate FILTER and MUST_NOT clauses`, which stores `MUST_NOT`
    ///    in a `HashSet` specifically so duplicates can never survive) and
    ///    is provably safe here: [`crate::matched_boolean_docs`] only ever
    ///    consumes `must_not` as a `Disjunction` used purely to *exclude*
    ///    docs matched by `must`/`should` (`Excluding::new(base, excluded)`),
    ///    and [`crate::clause_scores`] never iterates `must_not` at all --
    ///    only `must.iter().chain(should.iter())`. A `must_not` clause
    ///    therefore never contributes to score and never changes the
    ///    "excluded if it matches at least one" test when repeated, so
    ///    removing duplicates changes neither matched docs nor scores.
    ///
    /// 10. **`FILTER` clauses that are also `MUST`, or that match every
    ///     document, are dropped.** Java: `if (filters.size() > 1 ||
    ///     clauseSets.get(MUST).isEmpty() == false) modified =
    ///     filters.remove(MatchAllDocsQuery.INSTANCE); modified |=
    ///     filters.removeAll(clauseSets.get(Occur.MUST));`. The `MUST`
    ///     duplicate always goes -- the conjunction already requires it and
    ///     the filter copy adds no score. The `MatchAllDocsQuery` half is
    ///     guarded, because dropping the *only* filter of a filter-only query
    ///     would turn "every document, scored 0" into "no positive clauses",
    ///     i.e. a `MatchNoDocsQuery`.
    ///
    /// 11. **A clause that is both `FILTER` and `SHOULD` becomes `MUST`.**
    ///     Required *and* scored is exactly what `MUST` means. The `FILTER`
    ///     copy is dropped and `minimumNumberShouldMatch` is decremented
    ///     (floored at zero), because the promoted clause no longer counts
    ///     toward the threshold.
    ///
    /// 12. **A single `MUST` `MatchAllDocsQuery` alongside filters becomes a
    ///     `ConstantScoreQuery`.** With one scoring clause that matches
    ///     everything, the score is a constant, so the required half collapses
    ///     to `ConstantScoreQuery(BooleanQuery(FILTER.., MUST_NOT..))` with the
    ///     `MatchAllDocsQuery`'s boost carried on top, and the `SHOULD` clauses
    ///     are re-attached around it.
    ///
    /// 13. **A lone `FILTER` clause becomes `BoostQuery(ConstantScoreQuery(q),
    ///     0)`** -- Java's single-clause optimization, `case FILTER: // no
    ///     scoring clauses, so return a score of 0`. This is the one rule that
    ///     changes the query's *type*, because no `BooleanQuery` shape means
    ///     "match these, score nothing".
    ///
    /// Rules 4, 5, 7, 8, 10, 11, 12 and 13 all read the `filter` bucket; rules
    /// 4/5/7/8 treat it exactly as they treat `must` (`BooleanClause`'s
    /// `isRequired()` is `MUST || FILTER`), except that rule 7's inlining
    /// re-labels an inner `MUST` as a parent `FILTER` when the outer clause was
    /// a filter.
    ///
    /// **`must`/`should` duplicate deduplication (Java's two "Deduplicate …
    /// clauses by summing up their boosts" blocks).** Implemented, and worth a
    /// note because an earlier version of this file declined to implement it
    /// and justified the gap by citing
    /// `Similarity.computeQueryTermWeight` — **which does not exist in Lucene
    /// 10.5.0**, the version this port targets. It is a later addition on
    /// Lucene `main`; the `c18-version-audit` sweep batch caught the citation.
    ///
    /// 10.5.0's rule is a pure structural transform with no `IndexSearcher`
    /// and no `Similarity` in sight, so nothing about it is out of reach here:
    /// unwrap each clause's `BoostQuery` chain multiplying the boosts, sum the
    /// boosts per distinct unwrapped query, and — only when that collapsed the
    /// count — rebuild the bucket with one clause per distinct query, wrapped
    /// in a `BoostQuery` iff its summed boost is not `1`. `SHOULD` is gated on
    /// `minimumNumberShouldMatch <= 1` (a threshold of 2 or more counts
    /// *clauses*, so collapsing two into one would change what the query
    /// means); `MUST` is ungated.
    ///
    /// What this changes is query **shape**, not scores: BM25 is linear in the
    /// clause sum, so `a a` scores the same as `a^2` — see
    /// [`crate::clause_scores`], which sums every `must`/`should` clause's own
    /// per-doc score. What does change is the clause count (and so
    /// `maxClauseCount` pressure and the scorer count) and the explain tree,
    /// which is why it is worth doing rather than worth skipping.
    ///
    /// Two deliberate divergences, both stated rather than silent:
    ///
    /// - **Clause order is this port's, and it is deterministic.** Java
    ///   rebuilds from a `HashMap.entrySet()`, whose iteration order is
    ///   unspecified; this port keeps first-seen order. A stable order is a
    ///   strict improvement for a rewrite whose output feeds `explain`.
    /// - **Java returns immediately after a rebuild** and reaches the rest of
    ///   `rewrite` on `IndexSearcher.rewrite`'s next pass (it loops to a
    ///   fixpoint). This port is a single bottom-up pass, so a rebuild here
    ///   re-enters [`Self::rewrite`] on the rebuilt query instead — which is
    ///   the same fixpoint, and is needed rather than cosmetic: summing
    ///   `a^0.5 a^0.5` back to a bare `a` can make a later rule
    ///   (e.g. rule 10's "drop a FILTER that is also a MUST") newly apply.
    ///   Termination is bounded because a rebuild strictly reduces the bucket's
    ///   clause count.
    pub fn rewrite(self) -> Clause {
        // Java distinguishes "no clauses at all" from "only prohibited
        // clauses" by their reason strings, and only the original clause lists
        // can tell them apart -- after the `MatchNoDocsQuery` drops below, a
        // query that *had* `SHOULD` clauses can look identical to one that
        // never did.
        let originally_empty = self.must.is_empty()
            && self.filter.is_empty()
            && self.should.is_empty()
            && self.must_not.is_empty();
        let mut must: Vec<Clause> = self.must.into_iter().map(Clause::rewrite).collect();
        let mut filter: Vec<Clause> = self.filter.into_iter().map(Clause::rewrite).collect();
        let mut should: Vec<Clause> = self.should.into_iter().map(Clause::rewrite).collect();
        let mut must_not: Vec<Clause> = self.must_not.into_iter().map(Clause::rewrite).collect();
        let mut minimum_should_match = self.minimum_should_match;

        // Rule 5: a `MatchNoDocsQuery` clause. Real `BooleanQuery.rewrite()`
        // returns the `MatchNoDocsQuery` itself the moment a `MUST`/`FILTER`
        // clause rewrites to one, and silently drops a `SHOULD`/`MUST_NOT` one
        // ("the clause can be safely ignored").
        if let Some(no_match) = must.iter().chain(filter.iter()).find_map(|c| match c {
            Clause::MatchNoDocs(q) => Some(q.clone()),
            _ => None,
        }) {
            return Clause::MatchNoDocs(no_match);
        }
        should.retain(|c| !matches!(c, Clause::MatchNoDocs(_)));
        must_not.retain(|c| !matches!(c, Clause::MatchNoDocs(_)));

        // Rules 1 and 2, Java's first two: `if (clauses.size() == 0) return new
        // MatchNoDocsQuery("empty BooleanQuery");` and `if (clauses.size() ==
        // clauseSets.get(Occur.MUST_NOT).size()) return new
        // MatchNoDocsQuery("pure negative BooleanQuery");`.
        //
        // Java tests these *before* recursing and reaches the post-recursion
        // case on `IndexSearcher.rewrite`'s next pass (it loops to a fixpoint).
        // This port's `rewrite` is a single bottom-up pass, so the test runs
        // after the drops above instead, which reaches Java's fixpoint in one
        // go: a query whose only `SHOULD` clause rewrote to `MatchNoDocsQuery`
        // *is* a pure negative query by the time that clause is gone.
        //
        // `filter` counts as a positive clause here, exactly as it does in
        // Java: the pure-negative test is `clauses.size() ==
        // clauseSets.get(MUST_NOT).size()`, which a `FILTER` clause fails. A
        // filter-only query matches (at score 0); it is not "pure negative".
        if must.is_empty() && filter.is_empty() && should.is_empty() {
            return Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason(if originally_empty {
                "empty BooleanQuery"
            } else {
                "pure negative BooleanQuery"
            }));
        }

        // Rule 6: flatten a nested *pure disjunction* out of `should`.
        // `BooleanQuery.rewrite()`: "Flatten nested disjunctions, this is
        // important for block-max WAND to perform well". Java's
        // `isPureDisjunction()` is `every clause is SHOULD &&
        // minimumNumberShouldMatch <= 1`, and the outer query must itself have
        // `minimumNumberShouldMatch <= 1` or the count would change meaning.
        if minimum_should_match <= 1 {
            let mut flattened: Vec<Clause> = Vec::with_capacity(should.len());
            for clause in should {
                match clause {
                    Clause::Boolean(inner)
                        if inner.must.is_empty()
                            && inner.must_not.is_empty()
                            && !inner.should.is_empty()
                            && inner.minimum_should_match <= 1 =>
                    {
                        flattened.extend(inner.should);
                    }
                    other => flattened.push(other),
                }
            }
            should = flattened;
        }

        // Rule 7: inline a required clause that is itself a `BooleanQuery` with
        // no `should` clauses -- `BooleanQuery.rewrite()`'s "Inline required /
        // prohibited clauses. This helps run filtered conjunctive queries more
        // efficiently by providing all clauses to the block-max AND scorer."
        // The inner query's `must` clauses are also required of the parent, and
        // its `must_not` clauses are also prohibited to the parent, so both
        // lift verbatim. Guarded on `inner.should.is_empty() &&
        // inner.minimum_should_match == 0`, exactly Java's guard: an inner
        // `should` list would lose its "at least one" floor when merged into a
        // parent that has its own.
        //
        // Java asserts here that the inner query is not a pure negation,
        // "because the inner BooleanQuery would have first rewritten to a
        // MatchNoDocsQuery if it only had prohibited clauses". That now holds
        // in this port too: the recursion above rewrote the inner query, rules
        // 1/2 turned a pure-negative or empty one into `Clause::MatchNoDocs`,
        // and rule 5 turned a `MatchNoDocs` in `must` into the whole query's
        // result. So an inner `BooleanQuery` reaching this arm always has a
        // non-empty `must`, and the extra `!inner.must.is_empty()` guard this
        // rule carried before rules 1/2 existed is now provably dead --
        // removed rather than left as a comforting no-op.
        //
        // Java inlines out of every *required* clause, which is `MUST` **or**
        // `FILTER` (`BooleanClause.isRequired()`), and re-labels while lifting:
        // an inner `MUST` under an outer `FILTER` becomes a parent `FILTER`,
        // because the outer clause's whole subtree is non-scoring. Inner
        // `FILTER`/`MUST_NOT` clauses keep their occur under either outer occur.
        {
            let mut inlined_must: Vec<Clause> = Vec::with_capacity(must.len());
            let mut inlined_filter: Vec<Clause> = Vec::with_capacity(filter.len());
            let mut lifted_must_not: Vec<Clause> = Vec::new();
            for clause in must {
                match clause {
                    Clause::Boolean(inner)
                        if inner.should.is_empty() && inner.minimum_should_match == 0 =>
                    {
                        inlined_must.extend(inner.must);
                        inlined_filter.extend(inner.filter);
                        lifted_must_not.extend(inner.must_not);
                    }
                    other => inlined_must.push(other),
                }
            }
            for clause in filter {
                match clause {
                    Clause::Boolean(inner)
                        if inner.should.is_empty() && inner.minimum_should_match == 0 =>
                    {
                        // `assert outerClause.occur() == Occur.FILTER &&
                        // innerOccur == Occur.MUST; // ... change the occur of
                        // the inner query from MUST to FILTER.`
                        inlined_filter.extend(inner.must);
                        inlined_filter.extend(inner.filter);
                        lifted_must_not.extend(inner.must_not);
                    }
                    other => inlined_filter.push(other),
                }
            }
            must = inlined_must;
            filter = inlined_filter;
            must_not.extend(lifted_must_not);
        }

        // Rule 4: drop exact duplicate `filter` and `must_not` clauses
        // (order-preserving, keeps the first occurrence) -- Java's "remove
        // duplicate FILTER and MUST_NOT clauses", which it gets for free
        // because `clauseSets` stores those two occurs in a `HashSet` while
        // `MUST`/`SHOULD` go into a `Multiset`. See the doc comment above for
        // why the same is *not* safe for `must`/`should`. Runs after the
        // inlining above so a duplicate lifted out of a nested query is caught
        // too.
        dedup_clauses(&mut filter);
        dedup_clauses(&mut must_not);

        // Rule 8: "Check whether some clauses are both required and excluded"
        // -- `MatchNoDocsQuery("FILTER or MUST clause also in MUST_NOT")`, and
        // `MatchNoDocsQuery("MUST_NOT clause is MatchAllDocsQuery")`. Java's
        // predicate is `clauseSets.get(MUST)::contains` **or**
        // `clauseSets.get(FILTER)::contains` -- a clause required only as a
        // filter is just as excluded by a matching `MUST_NOT`.
        if must_not
            .iter()
            .any(|c| must.contains(c) || filter.contains(c))
        {
            return Clause::MatchNoDocs(
                MatchNoDocsQuery::new().with_reason("FILTER or MUST clause also in MUST_NOT"),
            );
        }
        if must_not
            .iter()
            .any(|c| matches!(c, Clause::MatchAllDocs(_)))
        {
            return Clause::MatchNoDocs(
                MatchNoDocsQuery::new().with_reason("MUST_NOT clause is MatchAllDocsQuery"),
            );
        }

        // Rule 10: "remove FILTER clauses that are also MUST clauses or that
        // match all documents". Java, verbatim:
        //
        //     if (filters.size() > 1 || clauseSets.get(MUST).isEmpty() == false) {
        //       modified = filters.remove(MatchAllDocsQuery.INSTANCE);
        //     }
        //     modified |= filters.removeAll(clauseSets.get(Occur.MUST));
        //
        // The guard on the `MatchAllDocsQuery` half is the subtle part and is
        // not an optimization detail: dropping the *only* filter of a
        // filter-only query would turn "every document, scored 0" into "no
        // positive clauses at all", i.e. a `MatchNoDocsQuery`. A
        // `MatchAllDocsQuery` filter is redundant only when something else
        // still constrains the match set.
        //
        // A `MUST` duplicate, by contrast, always goes: the conjunction
        // already requires it, and the filter copy adds no score, so it is
        // pure work. (This is the direction that matters -- the `MUST` copy is
        // kept, not the `FILTER` one, because the `MUST` copy is the scoring
        // one.)
        if !filter.is_empty() {
            if filter.len() > 1 || !must.is_empty() {
                filter.retain(|c| !matches!(c, Clause::MatchAllDocs(_)));
            }
            filter.retain(|c| !must.contains(c));
        }

        // Rule 11: "convert FILTER clauses that are also SHOULD clauses to MUST
        // clauses". A clause that is both required-but-unscored and
        // optional-and-scored is exactly a `MUST` clause: required, and
        // scored. Java drops the `FILTER` copy, promotes the `SHOULD` copy to
        // `MUST`, and decrements `minimumNumberShouldMatch` (floored at zero)
        // because that `SHOULD` no longer counts toward the threshold.
        if !should.is_empty() && !filter.is_empty() {
            let promoted: Vec<Clause> = should
                .iter()
                .filter(|c| filter.contains(c))
                .cloned()
                .collect();
            if !promoted.is_empty() {
                should.retain(|c| !filter.contains(c));
                filter.retain(|c| !promoted.contains(c));
                minimum_should_match = minimum_should_match.saturating_sub(promoted.len());
                must.extend(promoted);
            }
        }

        // Java's two `Deduplicate <occur> clauses by summing up their
        // boosts` blocks, in Java's own order (SHOULD then MUST, both after
        // rule 11 and before rule 12). See this method's doc comment for the
        // gating, the ordering divergence and why a rebuild re-enters
        // `rewrite` rather than falling through.
        let mut deduplicated = false;
        if !should.is_empty() && minimum_should_match <= 1 {
            deduplicated |= dedup_by_boost_sum(&mut should);
        }
        if !must.is_empty() {
            deduplicated |= dedup_by_boost_sum(&mut must);
        }
        if deduplicated {
            return BooleanQuery {
                must,
                filter,
                should,
                must_not,
                minimum_should_match,
            }
            .rewrite();
        }

        // Rule 12: "Rewrite queries whose single scoring clause is a MUST
        // clause on a MatchAllDocsQuery to a ConstantScoreQuery". With exactly
        // one `MUST`, and it matches everything, the conjunction's score is a
        // constant -- so the whole required half becomes a
        // `ConstantScoreQuery` over the filters and prohibitions, and the
        // `SHOULD` clauses are re-attached around it. A `BoostQuery` wrapper on
        // the `MatchAllDocsQuery` carries its boost onto the constant.
        if must.len() == 1 && !filter.is_empty() {
            let (inner_must, boost) = match &must[0] {
                Clause::Boost(b) => (b.inner.as_ref(), b.boost),
                other => (other, 1.0f32),
            };
            if matches!(inner_must, Clause::MatchAllDocs(_)) {
                let required = Clause::Boolean(Box::new(BooleanQuery {
                    must: Vec::new(),
                    filter,
                    should: Vec::new(),
                    must_not,
                    minimum_should_match: 0,
                }));
                let mut rewritten = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                    required,
                    // `ConstantScoreQuery`'s own score is its weight's boost,
                    // which is `1` here; the `BoostQuery` below applies the
                    // `MatchAllDocsQuery`'s boost on top, exactly as Java's
                    // `new BoostQuery(new ConstantScoreQuery(rewritten), boost)`
                    // does.
                    1.0,
                )));
                if boost != 1.0 {
                    rewritten = Clause::Boost(Box::new(BoostQuery::new(rewritten, boost)));
                }
                // Java re-attaches the `SHOULD` clauses around the constant and
                // returns that `BooleanQuery`; when there are none, its next
                // `IndexSearcher.rewrite` pass unwraps the resulting
                // single-`MUST` query. This port is a single bottom-up pass, so
                // it takes the fixpoint directly.
                if should.is_empty() {
                    return rewritten;
                }
                return Clause::Boolean(Box::new(BooleanQuery {
                    must: vec![rewritten],
                    filter: Vec::new(),
                    should,
                    must_not: Vec::new(),
                    minimum_should_match,
                }));
            }
        }

        // Rule 9: `should.len()` against `minimum_should_match`. Java runs both
        // halves of this *after* flattening, for the reason its own comment
        // gives ("this can only be processed after nested clauses have been
        // flattened"), which is why rule 6 above comes first.
        if should.len() < minimum_should_match {
            return Clause::MatchNoDocs(
                MatchNoDocsQuery::new()
                    .with_reason("SHOULD clause count less than minimumNumberShouldMatch"),
            );
        }
        if !should.is_empty() && should.len() == minimum_should_match {
            // Every `should` clause is required, which is what `MUST` means --
            // and `clause_scores` sums `must` and `should` identically, so the
            // scores are unchanged too.
            must.append(&mut should);
            minimum_should_match = 0;
            // Java returns `builder.build()` here and lets
            // `IndexSearcher.rewrite` loop; re-entering is that loop, and it
            // is load-bearing rather than tidy: this promotion runs *after*
            // the two dedup blocks, and the clauses it moves into `must` were
            // exempt from SHOULD dedup precisely because
            // `minimumNumberShouldMatch > 1` -- which is no longer true of
            // them. `+cat +cat` must still collapse to `cat^2`. Terminates
            // because `should` is now empty and the threshold is zero, so this
            // arm cannot fire again.
            return BooleanQuery {
                must,
                filter,
                should,
                must_not,
                minimum_should_match,
            }
            .rewrite();
        }

        // Rule 1, the single-clause unwrap. Java runs this *before* recursing
        // (`if (clauses.size() == 1)`), which is why every arm here also
        // requires the other three buckets to be empty -- "one clause" means
        // one clause in the whole query, not one in its bucket.
        if must_not.is_empty()
            && filter.is_empty()
            && minimum_should_match == 0
            && must.len() == 1
            && should.is_empty()
        {
            return must.into_iter().next().expect("len checked above");
        }
        if must_not.is_empty()
            && filter.is_empty()
            && minimum_should_match <= 1
            && should.len() == 1
            && must.is_empty()
        {
            return should.into_iter().next().expect("len checked above");
        }
        // Java: `case FILTER: // no scoring clauses, so return a score of 0
        // return new BoostQuery(new ConstantScoreQuery(query), 0);`. A lone
        // filter clause matches its documents at score `0` -- the one place
        // where a rewrite changes the query's *type*, because there is no
        // `BooleanQuery` shape that means "match these, score nothing".
        if must_not.is_empty()
            && must.is_empty()
            && should.is_empty()
            && minimum_should_match == 0
            && filter.len() == 1
        {
            let only = filter.into_iter().next().expect("len checked above");
            return Clause::Boost(Box::new(BoostQuery::new(
                Clause::ConstantScore(Box::new(ConstantScoreQuery::new(only, 1.0))),
                0.0,
            )));
        }

        Clause::Boolean(Box::new(BooleanQuery {
            must,
            filter,
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

    /// `BooleanQuery.Builder.add(query, Occur.FILTER)`: required for matching,
    /// contributing nothing to the score. See [`Self::filter`] for the exact
    /// semantics and [`Self::with_must`] for the accepted clause shapes.
    pub fn with_filter(mut self, clauses: impl IntoIterator<Item = impl Into<Clause>>) -> Self {
        self.filter.extend(clauses.into_iter().map(Into::into));
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
/// `i` (exact adjacency); with `slop > 0` a doc matches iff some choice of one
/// occurrence per slot fits in a window of width `slop` once each slot's
/// positions are shifted back by the slot's own index — which admits terms
/// appearing **out of phrase order**, exactly as real
/// `SloppyPhraseMatcher` does. See [`crate::sloppy_phrase`] for the ported
/// walk, and `docs/parity.md` for what is still out of scope.
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

/// `MultiPhraseQuery`-equivalent (`org.apache.lucene.search.MultiPhraseQuery`):
/// a phrase where every position accepts **any one of a set of terms** rather
/// than exactly one term -- the query a synonym filter or a prefix-expanded
/// last word produces (`"quick brown (fox|foxes)"`).
///
/// Scoped exactly like [`PhraseQuery`] is, and for the same reasons: positions
/// are implicitly `0, 1, ..., term_arrays.len() - 1` (real
/// `MultiPhraseQuery.Builder.add(Term[], int position)` allows explicit,
/// non-consecutive positions), and `slop > 0` runs the same ported
/// `SloppyPhraseMatcher` walk [`crate::sloppy_phrase`] documents -- including
/// its `hasMultiTermRpts` repeat handling, which only a `MultiPhraseQuery` can
/// reach.
///
/// **Semantics, taken from `MultiPhraseWeight`, not guessed:**
///
/// - **Matching.** Each position's alternatives behave as one merged posting
///   list (`MultiPhraseQuery.UnionFullPostingsEnum`): a document matches when
///   some alignment picks, at every position, an occurrence of *any* term in
///   that position's set. A term that is absent from the segment simply
///   contributes nothing to its position's union; a position whose whole set is
///   absent can never be satisfied, so the query matches nothing.
/// - **Scoring.** `MultiPhraseWeight` collects `TermStats` for **every** term
///   of **every** position and hands them all to `Similarity.scorer(...)`, so
///   the idf is the sum over all present terms -- not per position, and not the
///   max. The frequency is the merged-position phrase frequency, i.e. exactly
///   what [`crate::phrase_freq_exact`]/[`crate::sloppy_phrase::sloppy_phrase_freq`]
///   compute over the unioned position lists.
/// - **Degenerate shapes.** `MultiPhraseQuery.rewrite` turns an empty
///   `term_arrays` into a `MatchNoDocsQuery`, and a single-position one into a
///   `BooleanQuery` of `SHOULD` `TermQuery`s -- this port's executor reproduces
///   both outcomes directly (see [`crate::search_multi_phrase_query_scored`]),
///   with no rewrite step required.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MultiPhraseQuery {
    pub field: String,
    /// One entry per phrase position; each entry is that position's set of
    /// accepted terms (`MultiPhraseQuery`'s `termArrays`). An empty inner set
    /// is a position nothing can satisfy, so the query matches nothing.
    pub term_arrays: Vec<Vec<Vec<u8>>>,
    /// Sloppy-matching budget -- see [`PhraseQuery::slop`].
    pub slop: u32,
}

impl MultiPhraseQuery {
    /// Builds an exact (`slop == 0`) multi-phrase query. Each element of
    /// `term_arrays` is one position's alternatives, in phrase order.
    pub fn new(
        field: impl Into<String>,
        term_arrays: impl IntoIterator<Item = impl IntoIterator<Item = impl Into<Vec<u8>>>>,
    ) -> Self {
        Self {
            field: field.into(),
            term_arrays: term_arrays
                .into_iter()
                .map(|alts| alts.into_iter().map(Into::into).collect())
                .collect(),
            slop: 0,
        }
    }

    /// Builder method setting `slop`, consistent with [`PhraseQuery::with_slop`].
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
    fn rewrite_of_a_must_clause_with_an_unreachable_minimum_should_match_is_match_no_docs() {
        // must=[cat], should=[], minimum_should_match=1 matches nothing -- no
        // `should` clause can ever reach the threshold. Collapsing to a bare
        // `cat` clause would silently turn "matches nothing" into "matches
        // whatever cat matches"; real Lucene instead reports it, via the same
        // `shoulds.size() < minimumNumberShouldMatch` rule.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_minimum_should_match(1);
        let rewritten = q.rewrite();
        let Clause::MatchNoDocs(reason) = rewritten else {
            panic!("expected MatchNoDocs, got {rewritten:?}");
        };
        assert_eq!(
            reason.reason.as_str(),
            "SHOULD clause count less than minimumNumberShouldMatch"
        );
    }

    #[test]
    fn rewrite_of_one_should_clause_with_minimum_should_match_two_is_match_no_docs() {
        // Real `BooleanQuery.rewrite()`: `shoulds.size() <
        // minimumNumberShouldMatch` -> `MatchNoDocsQuery("SHOULD clause count
        // less than minimumNumberShouldMatch")`. This port used to leave the
        // query alone and rely on `matched_boolean_docs` reaching the same
        // "matches nothing" outcome at execution time -- same answer, but the
        // rewrite could not report it, and a caller inspecting the rewritten
        // query saw a query that looked satisfiable.
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat")])
            .with_minimum_should_match(2);
        let rewritten = q.rewrite();
        let Clause::MatchNoDocs(reason) = rewritten else {
            panic!("expected MatchNoDocs, got {rewritten:?}");
        };
        assert_eq!(
            reason.reason.as_str(),
            "SHOULD clause count less than minimumNumberShouldMatch"
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
                filter: Vec::new(),
                should: vec![],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_of_a_pure_must_not_only_query_is_match_no_docs() {
        // `BooleanQuery.rewrite()`: `clauses.size() ==
        // clauseSets.get(Occur.MUST_NOT).size()` -> `new
        // MatchNoDocsQuery("pure negative BooleanQuery")`. It is *not* "the
        // must_not clause itself", and it is no longer left structurally
        // intact -- see rule 2's doc comment for why that changed.
        let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(reason.reason.as_str(), "pure negative BooleanQuery");
    }

    #[test]
    fn rewrite_of_a_query_left_pure_negative_by_its_own_recursion_is_match_no_docs() {
        // The case Java only reaches on `IndexSearcher.rewrite`'s *second*
        // pass: the sole `should` clause is a nested query that itself
        // rewrites to `MatchNoDocsQuery`, rule 5 drops it, and what is left is
        // a pure negative query. This port's single bottom-up pass has to
        // catch it in one go, and this is the test that says so.
        let doomed = BooleanQuery::new().with_must_not([TermQuery::new("body", "cat")]);
        let q = BooleanQuery::new()
            .with_should([Clause::from(doomed)])
            .with_must_not([TermQuery::new("body", "dog")]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "pure negative BooleanQuery",
            "it had a SHOULD clause originally, so it is not the *empty* case"
        );
    }

    #[test]
    fn rewrite_removes_exact_duplicate_must_not_clauses() {
        // Rule 4: duplicate `must_not` clauses collapse to one, mirroring real
        // `BooleanQuery.rewrite()`'s "remove duplicate FILTER and MUST_NOT
        // clauses" step (`MUST_NOT` is stored in a `HashSet` there). Provably
        // safe: `must_not` only ever excludes docs (never scores), so
        // repeating a clause changes neither which docs match nor scores.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "dog"),
            ]);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_keeps_distinct_must_not_clauses_and_preserves_order() {
        // Non-duplicate `must_not` clauses are untouched, and dedup keeps the
        // *first* occurrence's position rather than reordering.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "fox"),
                TermQuery::new("body", "dog"),
            ]);
        let rewritten = q.clone().rewrite();
        assert_eq!(
            rewritten,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![
                    Clause::Term(TermQuery::new("body", "dog")),
                    Clause::Term(TermQuery::new("body", "fox")),
                ],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_deduplicates_must_not_clauses_inside_a_nested_boolean_clause() {
        // Rule 4 applies per-`BooleanQuery` level via the ordinary recursive
        // rewrite (rule 3), and the deduped nested query is then *inlined*
        // into its parent by rule 7 ("Inline required / prohibited clauses"),
        // because it is a required clause with no `should` list -- exactly
        // what real `BooleanQuery.rewrite()` does to feed every clause to one
        // conjunction scorer instead of nesting two.
        let inner = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "dog"), TermQuery::new("body", "dog")]);
        let outer = BooleanQuery::new()
            .with_must([Clause::from(inner)])
            .with_should([TermQuery::new("body", "bird")]);
        assert_eq!(
            outer.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                filter: Vec::new(),
                should: vec![Clause::Term(TermQuery::new("body", "bird"))],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_deduplicates_must_not_inside_a_nested_clause_that_cannot_be_inlined() {
        // Same per-level dedup, but on a nested query rule 7 must decline to
        // inline (it has its own `should` list, whose "at least one" floor
        // would be lost if merged into the parent's) -- so the nesting
        // survives and the dedup is visible at the inner level.
        let inner = BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "bird"),
            ])
            .with_must_not([TermQuery::new("body", "dog"), TermQuery::new("body", "dog")]);
        let outer = BooleanQuery::new()
            .with_must([Clause::from(inner)])
            .with_should([TermQuery::new("body", "fish")]);
        assert_eq!(
            outer.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Boolean(Box::new(BooleanQuery {
                    must: vec![],
                    filter: Vec::new(),
                    should: vec![
                        Clause::Term(TermQuery::new("body", "cat")),
                        Clause::Term(TermQuery::new("body", "bird")),
                    ],
                    must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                    minimum_should_match: 0,
                }))],
                filter: Vec::new(),
                should: vec![Clause::Term(TermQuery::new("body", "fish"))],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_of_a_match_no_docs_must_clause_is_the_whole_querys_result() {
        // Rule 5, the `MUST` half. `BooleanQuery.rewrite()`: when a clause
        // rewrites to `MatchNoDocsQuery`, `case MUST: case FILTER: return
        // rewritten;` -- the *inner* query is returned, reason string and all,
        // not a fresh one.
        let q = BooleanQuery::new()
            .with_must([
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("nothing here")),
            ])
            .with_should([TermQuery::new("body", "bird")]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "nothing here",
            "the offending clause's own reason must survive, not be replaced"
        );
    }

    #[test]
    fn rewrite_drops_a_match_no_docs_clause_from_should_and_must_not() {
        // Rule 5, the other half: `case SHOULD: case MUST_NOT: // the clause
        // can be safely ignored; break;`. A `SHOULD` that can never match
        // contributes nothing to the union, and a `MUST_NOT` that can never
        // match excludes nothing.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_should([
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("ignored")),
            ])
            .with_must_not([
                Clause::Term(TermQuery::new("body", "dog")),
                Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("also ignored")),
            ]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                filter: Vec::new(),
                should: vec![Clause::Term(TermQuery::new("body", "bird"))],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_flattens_a_nested_pure_disjunction_out_of_should() {
        // Rule 6. `BooleanQuery.rewrite()`: "Flatten nested disjunctions, this
        // is important for block-max WAND to perform well." Java's
        // `isPureDisjunction()` is "every clause is SHOULD and
        // minimumNumberShouldMatch <= 1", and the *outer* query must also have
        // `minimumNumberShouldMatch <= 1` or the count would change meaning.
        let inner = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let q = BooleanQuery::new()
            .with_should([
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::from(inner),
            ])
            .with_must([TermQuery::new("body", "fish")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "fish"))],
                filter: Vec::new(),
                should: vec![
                    Clause::Term(TermQuery::new("body", "bird")),
                    Clause::Term(TermQuery::new("body", "cat")),
                    Clause::Term(TermQuery::new("body", "dog")),
                ],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_does_not_flatten_a_nested_disjunction_that_is_not_pure() {
        // The three ways `isPureDisjunction()` fails, each of which must leave
        // the nesting alone: the inner query has a `must`, has a `must_not`,
        // or carries a `minimumNumberShouldMatch > 1`.
        let impure = [
            BooleanQuery::new()
                .with_should([TermQuery::new("body", "cat")])
                .with_must([TermQuery::new("body", "dog")]),
            BooleanQuery::new()
                .with_should([
                    TermQuery::new("body", "cat"),
                    TermQuery::new("body", "bird"),
                ])
                .with_must_not([TermQuery::new("body", "dog")]),
            BooleanQuery::new()
                .with_should([
                    TermQuery::new("body", "cat"),
                    TermQuery::new("body", "bird"),
                ])
                .with_minimum_should_match(2),
        ];
        for inner in impure {
            // The inner query is still *recursively rewritten* (rule 3) -- the
            // third case's `minimumNumberShouldMatch == 2` over two `SHOULD`
            // clauses becomes two `MUST` clauses via rule 9, for instance.
            // What must not happen is its clauses being lifted into the
            // parent, so the expectation is "still one nested clause, whatever
            // rewriting did to its insides".
            let expected_inner = Clause::from(inner.clone()).rewrite();
            let q = BooleanQuery::new().with_should([
                Clause::Term(TermQuery::new("body", "fish")),
                Clause::from(inner),
            ]);
            let Clause::Boolean(rewritten) = q.rewrite() else {
                panic!("expected a BooleanQuery");
            };
            assert_eq!(
                rewritten.should,
                vec![Clause::Term(TermQuery::new("body", "fish")), expected_inner],
                "a non-pure disjunction must not be flattened"
            );
        }
    }

    #[test]
    fn rewrite_does_not_flatten_when_the_outer_minimum_should_match_exceeds_one() {
        // Flattening a 1-clause `SHOULD` into 2 changes what "at least 2 of
        // the should clauses" means, so Java guards the whole block on
        // `minimumNumberShouldMatch <= 1`.
        let inner = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        // Three `SHOULD` clauses against a minimum of 2, so rule 9's
        // "every should is required" case does not also fire and the only
        // thing under test is the flattening guard.
        let q = BooleanQuery::new()
            .with_should([
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::Term(TermQuery::new("body", "fish")),
                Clause::from(inner.clone()),
            ])
            .with_minimum_should_match(2);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                filter: Vec::new(),
                should: vec![
                    Clause::Term(TermQuery::new("body", "bird")),
                    Clause::Term(TermQuery::new("body", "fish")),
                    Clause::from(inner),
                ],
                must_not: vec![],
                minimum_should_match: 2,
            }))
        );
    }

    #[test]
    fn rewrite_of_a_clause_that_is_both_required_and_excluded_is_match_no_docs() {
        // Rule 8, first half. `BooleanQuery.rewrite()`: "Check whether some
        // clauses are both required and excluded" -> `new
        // MatchNoDocsQuery("FILTER or MUST clause also in MUST_NOT")`. Both
        // halves of Java's predicate are now reachable
        // (`clauseSets.get(MUST)::contains` **or**
        // `clauseSets.get(FILTER)::contains`), so the reason string is Java's
        // verbatim rather than the shortened one this port used while it had
        // no `FILTER`.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "cat")]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "FILTER or MUST clause also in MUST_NOT"
        );

        // The `FILTER` half of the same predicate: a clause required only as a
        // filter is just as excluded by a matching `MUST_NOT`.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "dog")])
            .with_filter([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "cat")]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "FILTER or MUST clause also in MUST_NOT"
        );
    }

    // ---- `Occur.FILTER` rewrite rules -------------------------------------

    #[test]
    fn rewrite_of_a_lone_filter_clause_is_a_zero_boosted_constant_score_query() {
        // Java's single-clause optimization: `case FILTER: // no scoring
        // clauses, so return a score of 0 -- return new BoostQuery(new
        // ConstantScoreQuery(query), 0);`. Pinned bit-for-bit against real
        // Lucene in `tests/bm25_scoring_fixtures.rs`
        // (`scoring.boolean.filter.single` scores docs 0 and 2 at exactly 0).
        let q = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                    TermQuery::new("body", "cat"),
                    1.0
                ))),
                0.0
            )))
        );
    }

    #[test]
    fn rewrite_of_a_filter_only_query_is_not_a_pure_negative_query() {
        // `clauses.size() == clauseSets.get(MUST_NOT).size()` is false for a
        // FILTER clause, so the pure-negative collapse does not apply. Two
        // filters so the single-clause rule above does not fire either.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        assert_eq!(
            q.clone().rewrite(),
            Clause::Boolean(Box::new(q)),
            "a filter-only query rewrites to itself, not to MatchNoDocs"
        );
    }

    #[test]
    fn rewrite_of_a_match_no_docs_filter_clause_collapses_the_whole_query() {
        // Java: `case MUST: case FILTER: return rewritten;` -- a required
        // clause that matches nothing makes the conjunction match nothing.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([Clause::MatchNoDocs(
                MatchNoDocsQuery::new().with_reason("filter reason"),
            )]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "filter reason",
            "the offending clause's own reason survives, as in Java"
        );
    }

    #[test]
    fn rewrite_removes_duplicate_filter_clauses() {
        // `clauseSets` holds FILTER (and MUST_NOT) in a `HashSet`, so
        // duplicates cannot survive. Unlike MUST/SHOULD dedup, this changes no
        // score: a filter contributes none.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ]);
        let Clause::Boolean(rewritten) = q.rewrite() else {
            panic!("expected a BooleanQuery");
        };
        assert_eq!(
            rewritten.filter,
            vec![
                Clause::Term(TermQuery::new("body", "dog")),
                Clause::Term(TermQuery::new("body", "bird")),
            ]
        );
    }

    #[test]
    fn rewrite_drops_a_filter_clause_that_is_also_a_must_clause() {
        // `filters.removeAll(clauseSets.get(Occur.MUST))`. The MUST copy is the
        // one kept, because it is the scoring one. Pinned against real Lucene
        // as `scoring.boolean.filter.dupmust`, whose recorded scores are
        // bit-identical to `scoring.boolean.must`.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_filter([TermQuery::new("body", "dog")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery::new().with_must([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "dog")
            ])))
        );
    }

    #[test]
    fn rewrite_drops_a_match_all_docs_filter_only_when_something_else_constrains() {
        // Dropped: there is a MUST clause, so the filter is pure overhead.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))]);
        assert_eq!(
            q.rewrite(),
            Clause::Term(TermQuery::new("body", "cat")),
            "with the redundant filter gone, the single-clause unwrap fires"
        );

        // Dropped: more than one filter, so the others still constrain.
        let q = BooleanQuery::new().with_filter([
            Clause::MatchAllDocs(MatchAllDocsQuery::new(100)),
            Clause::Term(TermQuery::new("body", "cat")),
        ]);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                    TermQuery::new("body", "cat"),
                    1.0
                ))),
                0.0
            ))),
            "one filter left, so the lone-filter rule takes over"
        );

        // NOT dropped: it is the only filter and there is no MUST clause, so
        // removing it would turn "every document, scored 0" into "matches
        // nothing". This is the guard `filters.size() > 1 ||
        // clauseSets.get(MUST).isEmpty() == false` exists for.
        let q = BooleanQuery::new()
            .with_filter([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))])
            .with_must_not([TermQuery::new("body", "dog")]);
        assert_eq!(
            q.clone().rewrite(),
            Clause::Boolean(Box::new(q)),
            "the sole MatchAllDocs filter must survive"
        );
    }

    #[test]
    fn rewrite_promotes_a_clause_that_is_both_filter_and_should_to_must() {
        // Required *and* scored is what MUST means. The FILTER copy goes and
        // `minimumNumberShouldMatch` drops by one, since the promoted clause no
        // longer counts toward the threshold.
        //
        // Java returns `+cat dog~1` from this rule and reaches the final form
        // on `IndexSearcher.rewrite`'s next pass, where rule 9
        // (`shoulds.size() == minimumNumberShouldMatch`) turns the surviving
        // `dog` into a `MUST` too. This port's single bottom-up pass runs rule 9
        // after this one, so it lands on that fixpoint directly.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_minimum_should_match(2);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery::new().with_must([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "dog")
            ])))
        );
    }

    #[test]
    fn rewrite_floors_minimum_should_match_at_zero_when_promoting_filters() {
        // `builder.setMinimumNumberShouldMatch(Math.max(0, minShouldMatch))`.
        let q = BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        let Clause::Boolean(rewritten) = q.rewrite() else {
            panic!("expected a BooleanQuery");
        };
        assert_eq!(rewritten.minimum_should_match, 0);
        assert_eq!(
            rewritten.must,
            vec![Clause::Term(TermQuery::new("body", "cat"))]
        );
        assert_eq!(
            rewritten.should,
            vec![Clause::Term(TermQuery::new("body", "dog"))]
        );
        assert!(rewritten.filter.is_empty());
    }

    #[test]
    fn rewrite_turns_a_single_match_all_docs_must_plus_filters_into_a_constant_score_query() {
        // "Rewrite queries whose single scoring clause is a MUST clause on a
        // MatchAllDocsQuery to a ConstantScoreQuery."
        let q = BooleanQuery::new()
            .with_must([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))])
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
        assert_eq!(
            q.rewrite(),
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                BooleanQuery::new()
                    .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
                1.0
            )))
        );
    }

    #[test]
    fn rewrite_carries_the_match_all_docs_boost_onto_the_constant_score_query() {
        // `if (boost != 1f) rewritten = new BoostQuery(rewritten, boost);`, and
        // the SHOULD clauses are re-attached around the result.
        let q = BooleanQuery::new()
            .with_must([Clause::Boost(Box::new(BoostQuery::new(
                Clause::MatchAllDocs(MatchAllDocsQuery::new(100)),
                3.0,
            )))])
            .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_should([TermQuery::new("body", "bird")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([Clause::Boost(Box::new(BoostQuery::new(
                        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                            BooleanQuery::new().with_filter([
                                TermQuery::new("body", "cat"),
                                TermQuery::new("body", "dog")
                            ]),
                            1.0
                        ))),
                        3.0
                    )))])
                    .with_should([TermQuery::new("body", "bird")])
            ))
        );
    }

    #[test]
    fn rewrite_relabels_an_inner_must_as_a_filter_when_inlining_into_a_filter_clause() {
        // Java: `assert outerClause.occur() == Occur.FILTER && innerOccur ==
        // Occur.MUST; // In this case we need to change the occur of the inner
        // query from MUST to FILTER.` Getting this wrong would make the inner
        // clause score, which is exactly what the outer FILTER forbids.
        let inner = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")])
            .with_must_not([TermQuery::new("body", "bird")]);
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "fish")])
            .with_filter([Clause::from(inner)]);
        let Clause::Boolean(rewritten) = q.rewrite() else {
            panic!("expected a BooleanQuery");
        };
        assert_eq!(
            rewritten.must,
            vec![Clause::Term(TermQuery::new("body", "fish"))],
            "the inner MUST must NOT have landed in the parent's scoring bucket"
        );
        assert_eq!(
            rewritten.filter,
            vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ]
        );
        assert_eq!(
            rewritten.must_not,
            vec![Clause::Term(TermQuery::new("body", "bird"))]
        );
    }

    #[test]
    fn rewrite_inlines_an_inner_querys_filter_clauses_into_a_must_clause() {
        // The other direction: an outer MUST over an inner BooleanQuery lifts
        // the inner FILTER clauses as parent FILTER clauses (they stay
        // non-scoring), and the inner MUST clauses as parent MUST clauses.
        let inner = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]);
        let q = BooleanQuery::new()
            .with_must([Clause::from(inner)])
            .with_should([TermQuery::new("body", "bird")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([TermQuery::new("body", "cat")])
                    .with_filter([TermQuery::new("body", "dog")])
                    .with_should([TermQuery::new("body", "bird")])
            ))
        );
    }

    #[test]
    fn rewrite_leaves_a_filter_alongside_a_must_alone() {
        // The negative case for every rule above: nothing about `+cat #dog` is
        // redundant, so `rewrite` must be a no-op on it.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]);
        assert_eq!(q.clone().rewrite(), Clause::Boolean(Box::new(q)));
    }

    #[test]
    fn rewrite_of_a_must_not_match_all_docs_is_match_no_docs() {
        // Rule 8, second half: `if (mustNotClauses.contains(MatchAllDocsQuery
        // .INSTANCE)) return new MatchNoDocsQuery("MUST_NOT clause is
        // MatchAllDocsQuery");` -- excluding every document leaves none.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))]);
        let Clause::MatchNoDocs(reason) = q.rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(
            reason.reason.as_str(),
            "MUST_NOT clause is MatchAllDocsQuery"
        );
    }

    #[test]
    fn rewrite_leaves_a_must_not_alone_when_it_only_resembles_a_must_clause() {
        // The negative case for rule 8's first half: the clauses must be
        // *equal*, not merely same-field or same-shape. Without this, a query
        // like `+body:cat -body:dog` would be destroyed.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_must_not([TermQuery::new("body", "dog")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![Clause::Term(TermQuery::new("body", "cat"))],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_turns_every_should_into_must_when_the_minimum_equals_their_count() {
        // Rule 9's second half: `if (shoulds.size() > 0 && shoulds.size() ==
        // minimumNumberShouldMatch)` -> every `SHOULD` becomes `MUST`. All of
        // them are required, which is what `MUST` means; `clause_scores` sums
        // `must` and `should` identically, so scores are unchanged too.
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
            .with_minimum_should_match(2);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![
                    Clause::Term(TermQuery::new("body", "cat")),
                    Clause::Term(TermQuery::new("body", "dog")),
                ],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn rewrite_collapses_duplicate_must_clauses_into_one_with_summed_boosts() {
        // 10.5.0's `Deduplicate MUST clauses by summing up their boosts`:
        // `+cat +cat` becomes one clause at boost 2 -- and then rule 1's
        // single-clause unwrap takes the `BooleanQuery` away entirely, which
        // is what Java's next `IndexSearcher.rewrite` pass would also do.
        let q = BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                2.0
            )))
        );
    }

    #[test]
    fn rewrite_sums_explicit_boosts_across_duplicate_should_clauses() {
        // `a^2 a^3` -> `a^5`. The sum is Java's `double` accumulation narrowed
        // once at the end, not three `f32` additions.
        let q = BooleanQuery::new().with_should([
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                2.0,
            ))),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                3.0,
            ))),
        ]);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                5.0
            )))
        );
    }

    #[test]
    fn a_summed_boost_of_exactly_one_leaves_no_boost_wrapper_behind() {
        // Java: `if (boost != 1f) query = new BoostQuery(query, boost);` -- so
        // `a^0.5 a^0.5` collapses to a *bare* `a`, not to `a^1`. This is also
        // the case that makes the re-entry after a rebuild load-bearing: the
        // unwrapped `a` is now equal to the FILTER clause, so rule 10 must get
        // another look at it.
        let q = BooleanQuery::new()
            .with_should([
                Clause::Boost(Box::new(BoostQuery::new(
                    Clause::Term(TermQuery::new("body", "cat")),
                    0.5,
                ))),
                Clause::Boost(Box::new(BoostQuery::new(
                    Clause::Term(TermQuery::new("body", "cat")),
                    0.5,
                ))),
            ])
            .with_must([TermQuery::new("body", "dog")]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([TermQuery::new("body", "dog")])
                    .with_should([Clause::Term(TermQuery::new("body", "cat"))])
            ))
        );
    }

    #[test]
    fn nested_boost_wrappers_multiply_before_the_boosts_are_summed() {
        // Java's `while (query instanceof BoostQuery) { boost *= ...; }`:
        // `(a^2)^3` contributes 6, not 3 and not 2.
        let nested = Clause::Boost(Box::new(BoostQuery::new(
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                2.0,
            ))),
            3.0,
        )));
        let q =
            BooleanQuery::new().with_should([nested, Clause::Term(TermQuery::new("body", "cat"))]);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                7.0
            )))
        );
    }

    #[test]
    fn should_dedup_is_skipped_when_minimum_should_match_counts_more_than_one() {
        // Java gates the SHOULD half on `minimumNumberShouldMatch <= 1`: the
        // threshold counts *clauses*, so collapsing two into one would change
        // what the query means. `mSM == 2` with two identical clauses stays a
        // two-clause query -- and, since `should.len() == mSM`, rule 9 then
        // promotes both to MUST, which is where they end up.
        let q = BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")])
            .with_minimum_should_match(2);
        assert_eq!(
            q.rewrite(),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                2.0
            )))
        );
    }

    #[test]
    fn distinct_clauses_are_left_exactly_as_they_were() {
        // Java rebuilds only `if (map.size() != clauseSets.get(occur).size())`
        // -- with no duplicates there is no rebuild, so no clause acquires a
        // `BoostQuery` wrapper and the order is untouched.
        let q = BooleanQuery::new().with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ]);
        assert_eq!(
            q.rewrite(),
            Clause::Boolean(Box::new(BooleanQuery::new().with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "dog"),
                TermQuery::new("body", "bird"),
            ])))
        );
    }

    #[test]
    fn duplicate_clauses_collapse_in_first_seen_order() {
        // Java rebuilds from a `HashMap.entrySet()`, whose order is
        // unspecified; this port keeps first-seen order, deliberately -- a
        // rewrite whose output feeds `explain` should be deterministic.
        let q = BooleanQuery::new().with_should([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog"),
        ]);
        let Clause::Boolean(rewritten) = q.rewrite() else {
            panic!("expected a BooleanQuery");
        };
        assert_eq!(
            rewritten.should,
            vec![
                Clause::Boost(Box::new(BoostQuery::new(
                    Clause::Term(TermQuery::new("body", "dog")),
                    2.0
                ))),
                Clause::Term(TermQuery::new("body", "cat")),
            ]
        );
    }

    #[test]
    fn rewrite_of_an_empty_boolean_query_is_match_no_docs() {
        // `BooleanQuery.rewrite()`'s very first line: `if (clauses.size() == 0)
        // return new MatchNoDocsQuery("empty BooleanQuery");`. The reason
        // string is what distinguishes this from the pure-negative case, and
        // only the *original* clause lists can tell them apart.
        let Clause::MatchNoDocs(reason) = BooleanQuery::new().rewrite() else {
            panic!("expected MatchNoDocs");
        };
        assert_eq!(reason.reason.as_str(), "empty BooleanQuery");
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
                filter: Vec::new(),
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
                filter: Vec::new(),
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
