//! A minimal query-string parser inspired by real Lucene's classic
//! `org.apache.lucene.queryparser.classic.QueryParser` (and its
//! `StandardQueryParser` sibling) -- **not** a port of either class. Those
//! are large, hand-written grammars (JavaCC-generated in the classic case)
//! covering range queries, configurable boolean-operator precedence,
//! per-field analyzers, fuzzy similarity tuning, and a great deal of
//! escaping-edge-case handling. This module is a from-scratch, deliberately
//! small parser that turns a hand-picked subset of that syntax straight into
//! this port's existing [`crate::query::Clause`] tree -- reusing the
//! already-differentially-verified `TermQuery`/`PhraseQuery`/`BooleanQuery`/
//! `WildcardQuery`/`PrefixQuery`/`FuzzyQuery`/`RegexpQuery`/`BoostQuery`
//! constructors (see `crates/lucene-search/src/query.rs`) rather than adding
//! any new query representation.
//!
//! ## Supported grammar (the exact subset this module parses)
//!
//! ```text
//! query       := (modifier? atom suffix?) (conjunction modifier? atom suffix?)*
//! conjunction := 'AND' | '&&' | 'OR' | '||'        (absence => default operator)
//! modifier    := '+' | '-' | 'NOT' | '!'           (absence => MOD_NONE)
//! atom        := group | term
//! group       := '(' query ')'
//! term        := (field ':')? termbody
//! field       := identifier (ASCII letters/digits/'_'/'-'/'.', 1+ chars)
//! termbody    := phrase | regexp | range | wordterm
//! phrase      := '"' char* '"'                     => Clause::Phrase
//! regexp      := '/' char* '/'                     => Clause::Regexp
//! range       := ('['|'{') bound 'TO' bound (']'|'}')  => Clause::PointsRange
//! wordterm    := bareword                          => Clause::Wildcard / Prefix / Term
//! suffix      := '^' float ('~' number)? | '~' number ('^' float)?
//! ```
//!
//! **Boolean operators: `+`/`-` *and* `AND`/`OR`/`NOT`, with real Lucene's
//! precedence rules.** This module is a port of
//! `QueryParserBase.addClause`'s truth table, `&&`/`||`/`!` aliases included:
//!
//! - `AND` (`&&`) retroactively makes the **preceding** clause required as
//!   well as the following one, unless that preceding clause was prohibited
//!   -- which is what makes `a AND b` come out as `+a +b` rather than
//!   `a +b`.
//! - `OR` (`||`) under the `And` default operator retroactively makes the
//!   preceding clause optional, so `a OR b` is `a b`, not `+a b`.
//! - `NOT` (`!`) is a *modifier*, identical to `-`; `a AND NOT b` is
//!   therefore `+a -b`.
//! - `+`/`-` are the same modifiers they always were, and win over the
//!   conjunction.
//! - The default operator (what an absent conjunction means) is
//!   [`DefaultOperator::Or`] -- real Lucene's default -- unless the caller
//!   uses [`parse_query_with_operator`] to select
//!   [`DefaultOperator::And`].
//!
//! `AND`/`OR`/`NOT` are recognized only as a **whole** uppercase bareword
//! token, exactly as JavaCC's tokenizer does with `<AND: ("AND"|"&&")>`
//! declared ahead of `<TERM>`: `ANDROID` and `NOTHING` are ordinary terms,
//! `and` is an ordinary term, and only a standalone `AND` is the operator.
//!
//! **A query whose clause list collapses to one entry with no modifier is
//! returned unwrapped**, not as a one-clause [`crate::query::BooleanQuery`] --
//! Java's `clauses.size() == 1 && firstQuery != null` shortcut.
//!
//! **Default field.** `parse_query`'s `default_field` parameter supplies the
//! field for a bare (no `field:` prefix) term/phrase/wildcard/etc. If a bare
//! term appears and no default field was given, parsing fails with
//! [`ParseError::MissingField`] rather than guessing -- there is no implicit
//! "search every field" behavior in this slice (real Lucene's `QueryParser`
//! always requires a default field for exactly this reason).
//!
//! **One level of explicit grouping**, via `(...)`, but nesting is not
//! artificially limited -- `parse_group` recurses through the same
//! `parse_query` entry point, so `((a AND-ish b) c)`-shaped nesting parses
//! fine to arbitrary depth (the "one level" simplification in the task
//! description refers to there being exactly one grouping construct, not a
//! depth cap).
//!
//! **Wildcard vs. prefix disambiguation**: a bareword containing `*`/`?`
//! becomes a [`crate::query::WildcardQuery`], *except* when the only
//! special character is a single trailing, unescaped `*` (no `?` anywhere,
//! no other `*`), which becomes the simpler [`crate::query::PrefixQuery`] --
//! mirroring real Lucene's own `QueryParser`, which emits a `PrefixQuery`
//! for exactly the `foo*` shape and a `WildcardQuery` for anything else with
//! wildcard syntax in it.
//!
//! **Fuzzy**: `term~` (no digits) requests the default edit distance
//! ([`crate::query::FuzzyQuery::new`]'s default, `max_edits == 2`); `term~N`
//! for `N` in `0..=2` requests that many edits explicitly (matching real
//! `FuzzyQuery`'s supported range --
//! `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE == 2`); `N > 2` is a
//! [`ParseError::InvalidFuzziness`], not silently clamped. A `~` after a
//! wildcard or prefix bareword is **ignored**, because real
//! `QueryParserBase.handleBareTokenQuery` tests `wildcard`/`prefix` before
//! `fuzzy`. The fuzzy term is never run through the analyzer, matching
//! `getFuzzyQuery`.
//!
//! **Phrase slop**: `"a b"~N` sets the phrase's slop, real
//! `QueryParserBase.handleQuotedTerm`'s `(int) Float.parseFloat(...)` --
//! so a fractional `~1.7` truncates to `1` and a bare `~` leaves the default
//! slop of `0`. (This module used to reject `"a b"~N` outright, since `~` was
//! only ever a fuzzy marker.)
//!
//! **Boost and slop in either order**: `term^2~1` and `term~1^2` both parse,
//! matching real Lucene's `[ <CARAT> boost [ fuzzySlop ] | fuzzySlop
//! [ <CARAT> boost ] ]` production. At most one of each.
//!
//! **Regexp**: `/pattern/` (Lucene's own regexp delimiter convention) builds
//! a [`crate::query::RegexpQuery`] from the text between the slashes
//! verbatim; a `\/` inside a regexp escapes a literal slash without ending
//! the pattern (this is the only escape this module's regexp lexing
//! recognizes -- `RegexpPattern::new`'s own `\`-escaping of its operators
//! happens later, when the clause is resolved against a segment, and is
//! untouched by this module).
//!
//! **Escaping**: inside a bareword, `\` followed by any byte means that byte
//! is never treated as a wildcard/operator character (even if it's
//! `* ? : ~ ^ ( ) " / + -` or whitespace). For `Term`/`Fuzzy`/`Prefix` results
//! the escape is stripped (the resulting text is the literal bytes). For a
//! `Wildcard` result the escape is deliberately preserved in the pattern
//! handed to [`crate::query::WildcardQuery`] rather than stripped here --
//! [`lucene_codecs::wildcard::WildcardPattern::new`] does its own
//! `\`-escape parsing at resolve time, so if this parser stripped the
//! backslash first, an escaped literal `*` mixed with a genuine unescaped
//! `*` elsewhere in the same bareword would become indistinguishable from a
//! real wildcard operator once resolved. Whether a bareword ends up a
//! `Wildcard` at all is decided from genuine (unescaped) operator counts
//! tracked during the initial scan, not by re-inspecting the escaped text
//! afterward.
//!
//! **Numeric range queries**: `field:[min TO max]` (task #64's addition) --
//! an `i64` range, parsed into [`Clause::PointsRange`]. Either bound may be
//! `*` for an open end (mapped to `i64::MIN`/`i64::MAX`, matching real
//! Lucene's own unbounded-range convention), and either bound may be a
//! negative decimal integer (e.g. `field:[-100 TO 0]`). The `TO` keyword is
//! matched case-sensitively (real classic `QueryParser`'s grammar requires
//! uppercase `TO` too). **Parsing only**: the resulting
//! [`crate::query::PointsRangeQuery`] is not yet resolved against a segment
//! by anything in this crate -- see that struct's doc comment for the exact
//! deferred scope.
//!
//! **Exclusive bounds** (`{`/`}`) are supported independently per side, same
//! as real classic `QueryParser`: `[` / `]` mean inclusive, `{` / `}` mean
//! exclusive, and the two may mix in one range (`field:[0 TO 100}`,
//! `field:{0 TO 100]`, `field:{0 TO 100}`). Since [`crate::query::PointsRangeQuery`]
//! only stores one inclusive `[min, max]` pair, an exclusive *literal* bound
//! is converted to the equivalent inclusive one at parse time (stepped by
//! one toward the other bound, saturating at `i64::MIN`/`i64::MAX`) --
//! mirroring real Lucene's own `QueryParserBase.getRangeQuery`, which does
//! the same adjustment before constructing its inclusive-only
//! `PointRangeQuery`. An exclusive `*` (open) bound is left unstepped, since
//! `*` means "no bound" rather than a literal value.
//!
//! ## Deliberately deferred (parse error, not silent misinterpretation)
//!
//! - **String/date range queries** (`field:[aaa TO zzz]` over a
//!   non-numeric/`TermRangeQuery`-shaped field) -- a `[min TO max]` whose
//!   bounds don't parse as a plain (optionally negative) decimal integer or
//!   `*` is a [`ParseError::InvalidRangeBound`], not a fallback to string
//!   comparison.
//! - **Boosting a group's boost multiplying inner boosts / any boost
//!   algebra beyond one flat `^N` per atom** -- a `^` after a `)` applies
//!   exactly the same single [`crate::query::BoostQuery`] wrap a term/phrase
//!   boost gets, nothing fancier.
//! - **Fuzziness `~` with a fractional similarity (e.g. `term~0.8`, the
//!   pre-4.0 float-similarity convention real Lucene still lexes and then
//!   rejects with "Fractional edit distances are not allowed!" for values
//!   `>= 1.0`)** -- only bare `~` or `~` followed by an integer `0..=2` is
//!   accepted on a *bareword*; a `~` followed by a decimal point is a
//!   [`ParseError::InvalidFuzziness`]. A fractional *phrase* slop is accepted
//!   and truncated, as Java does.
//! - **`MultiFieldQueryParser`** -- one default field only; no
//!   fan-out-across-fields rewriting.
//! - **A leading `*`/`?` guard.** Real `QueryParserBase` throws
//!   `"'*' or '?' not allowed as first character in WildcardQuery"` unless
//!   `setAllowLeadingWildcard(true)`; this module always allows it, since
//!   [`lucene_codecs::wildcard::WildcardPattern`] has no
//!   pathological-cost cliff that guard exists to protect against here.
//! - **Any escaping edge case beyond the single `\`-then-any-byte rule
//!   above** (e.g. Unicode `\uXXXX` escapes, which real
//!   `QueryParserBase.escape` doesn't even round-trip for parsing).

use crate::query::{
    BooleanQuery, BoostQuery, Clause, FuzzyQuery, PhraseQuery, PointsRangeQuery, PrefixQuery,
    RegexpQuery, TermQuery, WildcardQuery,
};
use lucene_analysis::Analyzer;

/// Errors this parser can return -- every malformed input documented in the
/// module doc's "deliberately deferred" section, plus basic
/// unclosed-delimiter/unexpected-character cases, surfaces as one of these
/// rather than a panic.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// The input was empty (after trimming whitespace) -- there is no
    /// well-defined "empty query" `Clause` to return.
    #[error("empty query")]
    EmptyQuery,
    /// A `"` was opened but never closed before the input ended.
    #[error("unclosed phrase quote starting at byte {0}")]
    UnclosedQuote(usize),
    /// A `/` regexp delimiter was opened but never closed.
    #[error("unclosed regexp starting at byte {0}")]
    UnclosedRegexp(usize),
    /// A `(` was opened but never matched by a `)`.
    #[error("unclosed parenthesis starting at byte {0}")]
    UnclosedParen(usize),
    /// A `)` appeared with no matching open `(`.
    #[error("unexpected ')' at byte {0}")]
    UnmatchedCloseParen(usize),
    /// A bare (no `field:` prefix) term/phrase/wildcard/etc. was found but
    /// `parse_query` was called with `default_field: None`.
    #[error("term at byte {0} has no field and no default field was given")]
    MissingField(usize),
    /// `~` was followed by something other than an optional plain integer
    /// (e.g. a decimal, or an integer outside `0..=2`).
    #[error("invalid fuzziness at byte {0}: {1}")]
    InvalidFuzziness(usize, String),
    /// `^` was followed by something that doesn't parse as a finite `f32`.
    #[error("invalid boost at byte {0}: {1}")]
    InvalidBoost(usize, String),
    /// Syntax this module explicitly does not support (see the module doc's
    /// "deliberately deferred" list).
    #[error("unsupported syntax at byte {0}: {1}")]
    UnsupportedSyntax(usize, String),
    /// A `field:[min TO max]`/`field:{min TO max}` range bound wasn't `*` and
    /// didn't parse as a plain (optionally negative) decimal `i64`, or the
    /// `TO` keyword was missing/misspelled, or the range wasn't closed by a
    /// matching `]`/`}`.
    #[error("invalid range at byte {0}: {1}")]
    InvalidRangeBound(usize, String),
    /// A character appeared where no valid token could start (e.g. a bare
    /// `:` with no preceding field name, or a stray `~`/`^` with no
    /// preceding term).
    #[error("unexpected character {1:?} at byte {0}")]
    UnexpectedChar(usize, char),
    /// The input ended mid-token where more input was expected (e.g. right
    /// after a `+`/`-` modifier, or a `field:` with nothing after the
    /// colon).
    #[error("unexpected end of input, expected {0}")]
    UnexpectedEnd(&'static str),
}

/// How a top-level clause combines into the enclosing [`BooleanQuery`] --
/// real `BooleanClause.Occur`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Occur {
    Must,
    Should,
    MustNot,
}

/// `QueryParser.Modifiers()`'s result: the `+`/`-`/`NOT`/`!` prefix (or its
/// absence) on one clause.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Modifier {
    /// `MOD_NONE`.
    None,
    /// `MOD_REQ` (`+`).
    Required,
    /// `MOD_NOT` (`-`, `NOT`, `!`).
    Not,
}

/// `QueryParser.Conjunction()`'s result: the `AND`/`&&`/`OR`/`||` token (or
/// its absence) *between* two clauses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Conjunction {
    /// `CONJ_NONE`.
    None,
    /// `CONJ_AND`.
    And,
    /// `CONJ_OR`.
    Or,
}

/// `QueryParserBase.setDefaultOperator` — how two adjacent clauses with no
/// explicit `AND`/`OR` between them combine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DefaultOperator {
    /// `QueryParserBase.OR_OPERATOR`, real Lucene's default: a bare clause is
    /// optional (`SHOULD`).
    #[default]
    Or,
    /// `QueryParserBase.AND_OPERATOR`: a bare clause is required (`MUST`)
    /// unless introduced by `OR`.
    And,
}

/// Parses `input` (classic-Lucene-inspired query-string syntax -- see this
/// module's doc comment for the exact supported grammar) into this port's
/// [`Clause`] tree. `default_field` supplies the field for any bare (no
/// `field:` prefix) term; `None` makes a bare term a
/// [`ParseError::MissingField`].
pub fn parse_query(input: &str, default_field: Option<&str>) -> Result<Clause, ParseError> {
    parse_query_with_analyzer(input, default_field, None)
}

/// Same as [`parse_query`], but when `analyzer` is `Some`, every bareword
/// term's text -- both plain `wordterm`s and each whitespace-separated word of
/// a quoted `phrase` -- is run through the analyzer before becoming a
/// [`Clause`], mirroring real Lucene's `QueryParser`, which analyzes query
/// text through the same `Analyzer` used at index time rather than treating
/// the raw query string as literal terms.
///
/// **Not analyzed**: wildcard (`Clause::Wildcard`/`Clause::Prefix`), fuzzy
/// (`Clause::Fuzzy`), and regexp (`Clause::Regexp`) pattern text. Real
/// Lucene's classic `QueryParser` does not analyze these either -- running an
/// analyzer (tokenization, lowercasing, stopword removal) over glob/regex
/// syntax would corrupt the pattern (e.g. splitting `c*t` into `c`/`t` tokens,
/// destroying the wildcard).
///
/// **Multi-token/zero-token handling** for an analyzed bareword or phrase
/// word (a simplification of real `QueryParserBase.newFieldQuery`'s fuller
/// multi-token handling, which additionally builds position-aware
/// `SynonymQuery`/graph queries in some cases -- out of scope here): if the
/// analyzer produces exactly one token, it becomes a single `Clause::Term`
/// (or, within a phrase, a single phrase position); if it produces zero
/// tokens (e.g. the bareword was itself a single stopword), the result is an
/// empty [`BooleanQuery`] (`must`/`should`/`must_not` all empty), which
/// already means "matches nothing" throughout this crate (see
/// `matched_boolean_docs`'s doc comment) -- a clean no-match rather than an
/// error; if it produces multiple tokens, they become a [`Clause::Phrase`] in
/// order (for a bareword) or are spliced in place (for one word of an
/// already-multi-word phrase).
pub fn parse_query_with_analyzer(
    input: &str,
    default_field: Option<&str>,
    analyzer: Option<&Analyzer>,
) -> Result<Clause, ParseError> {
    parse_query_with_operator(input, default_field, analyzer, DefaultOperator::Or)
}

/// Same as [`parse_query_with_analyzer`], but with an explicit
/// [`DefaultOperator`] — real `QueryParserBase.setDefaultOperator`. `Or` (real
/// Lucene's default, what the other two entry points use) makes a clause with
/// no `+`/`-`/`AND`/`OR` optional; `And` makes it required unless an `OR`
/// introduces it.
pub fn parse_query_with_operator(
    input: &str,
    default_field: Option<&str>,
    analyzer: Option<&Analyzer>,
    default_operator: DefaultOperator,
) -> Result<Clause, ParseError> {
    let bytes: Vec<char> = input.chars().collect();
    let mut parser = Parser {
        chars: &bytes,
        pos: 0,
        default_field,
        analyzer,
        default_operator,
        multi_fields: &[],
        used_default_field: false,
    };
    parser.skip_ws();
    if parser.pos >= parser.chars.len() {
        return Err(ParseError::EmptyQuery);
    }
    // `parse_clause_list(false)` only returns `Ok` once it has consumed every
    // remaining character (its loop only exits normally on `peek() == None`;
    // a `')'` at the top level is rejected inside the loop itself, before
    // ever returning `Ok`) -- so there is no "trailing unparsed input" case
    // to check for here.
    parser.parse_clause_list(false)
}

/// `MultiFieldQueryParser(fields, analyzer)` -- parses `input` fanning every
/// **bare** (unqualified) atom out across `fields` as a `SHOULD` disjunction.
///
/// An atom that names its own field (`title:cat`) is left alone, exactly as
/// Java's `field != null` branch does. `title:cat dog` over
/// `["title", "body"]` is `title:cat (title:dog body:dog)`.
///
/// # Errors
///
/// [`ParseError::MissingField`] never fires in multi-field mode (there is
/// always a field), but `fields` must be non-empty -- an empty `fields` is
/// [`ParseError::MissingField`] at byte 0, since every bare atom would then
/// have nowhere to go. A single-element `fields` is exactly
/// [`parse_query_with_analyzer`] with that default field.
pub fn parse_multi_field_query(
    input: &str,
    fields: &[&str],
    analyzer: Option<&Analyzer>,
) -> Result<Clause, ParseError> {
    let with_boosts: Vec<(&str, f32)> = fields.iter().map(|f| (*f, 1.0)).collect();
    parse_multi_field_query_with_boosts(input, &with_boosts, analyzer, DefaultOperator::Or)
}

/// `MultiFieldQueryParser(fields, analyzer, boosts)` plus
/// `setDefaultOperator` -- the full form.
///
/// Each `(field, boost)` pair fans a bare atom out to `field`, wrapping that
/// field's clause in a [`BoostQuery`] when `boost != 1.0` (Java looks the
/// boost up in its `Map<String, Float>` and skips the wrap when the map has
/// no entry; a `1.0` here is the same no-op, since a `BoostQuery` of `1.0`
/// multiplies every score by one).
///
/// With [`DefaultOperator::And`], `cat dog` over `title`/`body` with boosts
/// `title => 5, body => 10` is Java's documented
/// `+(title:cat^5.0 body:cat^10.0) +(title:dog^5.0 body:dog^10.0)`.
pub fn parse_multi_field_query_with_boosts(
    input: &str,
    fields: &[(&str, f32)],
    analyzer: Option<&Analyzer>,
    default_operator: DefaultOperator,
) -> Result<Clause, ParseError> {
    if fields.is_empty() {
        return Err(ParseError::MissingField(0));
    }
    let owned: Vec<(String, f32)> = fields.iter().map(|(f, b)| ((*f).to_string(), *b)).collect();
    let chars: Vec<char> = input.chars().collect();
    let mut parser = Parser {
        chars: &chars,
        pos: 0,
        default_field: Some(&owned[0].0),
        analyzer,
        default_operator,
        multi_fields: &owned,
        used_default_field: false,
    };
    parser.skip_ws();
    if parser.pos >= parser.chars.len() {
        return Err(ParseError::EmptyQuery);
    }
    parser.parse_clause_list(false)
}

/// `MultiFieldQueryParser.applyBoost`: wraps `clause` in a [`BoostQuery`]
/// when this field has a boost configured. `1.0` is Java's "no entry in the
/// boosts map" -- left unwrapped, since the query is identical either way.
fn apply_field_boost(clause: Clause, boost: f32) -> Clause {
    if boost == 1.0 {
        clause
    } else {
        Clause::Boost(Box::new(BoostQuery::new(clause, boost)))
    }
}

/// Runs `text` through `analyzer` (if any), returning the resulting term
/// strings in order. `None` means "no analysis" -- `text` passes through
/// unchanged as a single term, preserving this parser's pre-analyzer literal
/// behavior exactly.
fn analyze_term_text(analyzer: Option<&Analyzer>, text: &str) -> Vec<String> {
    match analyzer {
        None => vec![text.to_string()],
        Some(analyzer) => analyzer.analyze(text).into_iter().map(|t| t.term).collect(),
    }
}

/// A [`Clause`] that matches no documents -- an empty [`BooleanQuery`] (no
/// `must`/`should`/`must_not` clauses), which `matched_boolean_docs` already
/// treats as `MatchNoDocsQuery` (see that function's doc comment). Used for
/// the zero-token case: a bareword or phrase that analyzed away to nothing
/// (e.g. it was itself a stopword) is a clean no-match, not an error.
fn no_match_clause() -> Clause {
    Clause::Boolean(Box::new(BooleanQuery {
        must: Vec::new(),
        filter: Vec::new(),
        should: Vec::new(),
        must_not: Vec::new(),
        minimum_should_match: 0,
    }))
}

/// Builds the [`Clause`] for one analyzed bareword, applying the
/// zero/one/multi-token handling documented on [`parse_query_with_analyzer`]:
/// zero tokens => [`no_match_clause`]; one token => `Clause::Term`; more than
/// one => `Clause::Phrase` in order.
fn clause_from_analyzed_terms(field: &str, mut terms: Vec<String>) -> Clause {
    match terms.len() {
        0 => no_match_clause(),
        1 => Clause::Term(TermQuery::new(field, terms.remove(0))),
        _ => Clause::Phrase(PhraseQuery::new(field, terms)),
    }
}

struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
    default_field: Option<&'a str>,
    analyzer: Option<&'a Analyzer>,
    default_operator: DefaultOperator,
    /// `MultiFieldQueryParser.fields` + `.boosts`, or empty for the
    /// single-field parser. See [`parse_multi_field_query_with_boosts`].
    multi_fields: &'a [(String, f32)],
    /// Set by [`Self::parse_term`]/[`Self::parse_atom`] whenever an atom fell
    /// back to `default_field` -- i.e. Java's `field == null` condition inside
    /// `MultiFieldQueryParser.getFieldQuery`. Cleared by
    /// [`Self::parse_boosted_atom`] once it has expanded that atom, so an
    /// enclosing group never expands a second time.
    used_default_field: bool,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let c = self.peek()?;
        self.pos += 1;
        Some(c)
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(c) if c.is_whitespace()) {
            self.pos += 1;
        }
    }

    /// `QueryParser.Query(field)`: `(Modifiers Clause) (Conjunction Modifiers
    /// Clause)*`, folded into a [`Clause`] by [`Self::add_clause`] --
    /// `QueryParserBase.addClause`'s exact `conj`/`mods`/`defaultOperator`
    /// truth table, including its retroactive rewrite of the *previous*
    /// clause when an `AND`/`OR` follows it.
    ///
    /// As in Java, a query whose clause list collapsed to exactly one entry
    /// that carried no modifier is returned unwrapped rather than as a
    /// one-clause [`BooleanQuery`] (`clauses.size() == 1 && firstQuery !=
    /// null`).
    fn parse_clause_list(&mut self, inside_group: bool) -> Result<Clause, ParseError> {
        let mut clauses: Vec<(Occur, Clause)> = Vec::new();
        let mut first_query: Option<Clause> = None;
        let mut first = true;

        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(')') if inside_group => break,
                Some(')') => return Err(ParseError::UnmatchedCloseParen(self.pos)),
                _ => {}
            }

            // `Conjunction()` -- never consumed before the first clause
            // (Java's grammar only allows one from the second iteration on).
            let conj = if first {
                Conjunction::None
            } else {
                self.parse_conjunction()
            };
            self.skip_ws();
            if self.peek().is_none() {
                return Err(ParseError::UnexpectedEnd("a clause after a conjunction"));
            }

            let mods = self.parse_modifiers();
            self.skip_ws();
            let clause = self.parse_boosted_atom()?;

            if first && mods == Modifier::None {
                first_query = Some(clause.clone());
            }
            if !first && first_query.is_some() {
                first_query = None;
            }
            self.add_clause(&mut clauses, conj, mods, clause);
            first = false;
        }

        if clauses.is_empty() {
            return Err(ParseError::EmptyQuery);
        }
        if clauses.len() == 1 {
            if let Some(clause) = first_query {
                return Ok(clause);
            }
        }

        let mut must = Vec::new();
        let mut should = Vec::new();
        let mut must_not = Vec::new();
        for (occur, clause) in clauses {
            match occur {
                Occur::Must => must.push(clause),
                Occur::Should => should.push(clause),
                Occur::MustNot => must_not.push(clause),
            }
        }
        Ok(Clause::Boolean(Box::new(BooleanQuery {
            must,
            // `QueryParserBase.addClause` never produces `Occur.FILTER`: its
            // three outcomes are MUST (`+`/`AND`), SHOULD and MUST_NOT
            // (`-`/`NOT`). Classic query syntax has no filter operator.
            filter: Vec::new(),
            should,
            must_not,
            minimum_should_match: 0,
        })))
    }

    /// `QueryParserBase.addClause`, verbatim -- including the two retroactive
    /// rewrites of the previous clause that make `a AND b` and (under
    /// `AND_OPERATOR`) `a OR b` come out right, and the operator-dependent
    /// `required`/`prohibited` derivation.
    fn add_clause(
        &self,
        clauses: &mut Vec<(Occur, Clause)>,
        conj: Conjunction,
        mods: Modifier,
        query: Clause,
    ) {
        // "If this term is introduced by AND, make the preceding term
        // required, unless it's already prohibited."
        if conj == Conjunction::And {
            if let Some(last) = clauses.last_mut() {
                if last.0 != Occur::MustNot {
                    last.0 = Occur::Must;
                }
            }
        }
        // "If this term is introduced by OR, make the preceding term
        // optional, unless it's prohibited" -- only under AND_OPERATOR, where
        // the preceding term would otherwise already be required.
        if self.default_operator == DefaultOperator::And && conj == Conjunction::Or {
            if let Some(last) = clauses.last_mut() {
                if last.0 != Occur::MustNot {
                    last.0 = Occur::Should;
                }
            }
        }

        let (required, prohibited) = match self.default_operator {
            DefaultOperator::Or => {
                let prohibited = mods == Modifier::Not;
                let required =
                    mods == Modifier::Required || (conj == Conjunction::And && !prohibited);
                (required, prohibited)
            }
            DefaultOperator::And => {
                let prohibited = mods == Modifier::Not;
                let required = !prohibited && conj != Conjunction::Or;
                (required, prohibited)
            }
        };
        let occur = match (required, prohibited) {
            (true, false) => Occur::Must,
            (false, false) => Occur::Should,
            // `required && prohibited` is unreachable: `required` is only ever
            // set when `prohibited` is false in both branches above, exactly
            // as Java's `throw new RuntimeException("Clause cannot be both
            // required and prohibited")` is unreachable there.
            (_, true) => Occur::MustNot,
        };
        clauses.push((occur, query));
    }

    /// `QueryParser.Conjunction()`: `AND` / `&&` / `OR` / `||`, or nothing.
    ///
    /// The word forms are recognized only when the whole bareword token is
    /// exactly `AND`/`OR` (uppercase, and not a prefix of a longer word) --
    /// the same thing JavaCC's longest-match-then-first-rule tokenizer does
    /// with `<AND: ("AND"|"&&")>` declared ahead of `<TERM>`, which is why
    /// `ANDROID` is a term and `AND` is an operator.
    fn parse_conjunction(&mut self) -> Conjunction {
        if self.try_consume_symbol("&&") {
            return Conjunction::And;
        }
        if self.try_consume_symbol("||") {
            return Conjunction::Or;
        }
        if self.try_consume_keyword("AND") {
            return Conjunction::And;
        }
        if self.try_consume_keyword("OR") {
            return Conjunction::Or;
        }
        Conjunction::None
    }

    /// `QueryParser.Modifiers()`: `+` / `-` / `NOT` / `!`, or nothing.
    fn parse_modifiers(&mut self) -> Modifier {
        match self.peek() {
            Some('+') => {
                self.advance();
                Modifier::Required
            }
            Some('-') => {
                self.advance();
                Modifier::Not
            }
            Some('!') => {
                self.advance();
                Modifier::Not
            }
            _ => {
                if self.try_consume_keyword("NOT") {
                    Modifier::Not
                } else {
                    Modifier::None
                }
            }
        }
    }

    /// Consumes `symbol` if it sits at the cursor. Used for `&&`/`||`, which
    /// -- unlike `AND`/`OR` -- need no word boundary after them.
    fn try_consume_symbol(&mut self, symbol: &str) -> bool {
        let chars: Vec<char> = symbol.chars().collect();
        if self.chars[self.pos..].starts_with(&chars) {
            self.pos += chars.len();
            true
        } else {
            false
        }
    }

    /// Consumes `keyword` if the *whole* bareword token at the cursor is
    /// exactly it -- i.e. the next character after it is not another bareword
    /// character. `AND`/`OR`/`NOT` are matched case-sensitively, as real
    /// `QueryParser`'s grammar does.
    fn try_consume_keyword(&mut self, keyword: &str) -> bool {
        let chars: Vec<char> = keyword.chars().collect();
        if !self.chars[self.pos..].starts_with(&chars) {
            return false;
        }
        let after = self.chars.get(self.pos + chars.len()).copied();
        // A bareword character right after means this is a longer term
        // ("ANDROID", "NOTHING"), not the operator.
        if let Some(c) = after {
            if !is_term_stop_char(c) || c == ':' {
                return false;
            }
        }
        self.pos += chars.len();
        true
    }

    /// `QueryParser.Term`'s suffix rule, applied to any atom:
    /// `[ '^' boost [ '~' n ] | '~' n [ '^' boost ] ]` -- real Lucene accepts
    /// the fuzziness/phrase-slop marker and the boost in **either** order, and
    /// at most one of each.
    fn parse_boosted_atom(&mut self) -> Result<Clause, ParseError> {
        let atom_start = self.pos;
        self.used_default_field = false;
        let clause = self.parse_atom_with_suffixes()?;
        if self.multi_fields.len() > 1 && self.used_default_field {
            self.used_default_field = false;
            return self.expand_across_fields(atom_start, clause);
        }
        self.used_default_field = false;
        Ok(clause)
    }

    /// `MultiFieldQueryParser.getMultiFieldQuery(queries)`: the same atom,
    /// once per configured field, joined as `SHOULD` clauses.
    ///
    /// Java expands inside `getFieldQuery(null, ...)` -- calling
    /// `super.getFieldQuery(fields[i], queryText, quoted)` per field and
    /// `builder.add(sub, Occur.SHOULD)` -- so the disjunction sits at the
    /// *leaf*, under the outer query's own conjunctions and modifiers.
    /// `cat AND dog` over `title`/`body` is therefore
    /// `+(title:cat body:cat) +(title:dog body:dog)`, which requires each term
    /// in *some* field, and not `+(title:cat title:dog) +(body:cat body:dog)`,
    /// which would require both terms in the same field. Re-parsing the atom's
    /// own span once per field reproduces that placement exactly.
    ///
    /// Two things about doing it by re-parsing rather than by rewriting the
    /// built clause:
    ///
    /// - **The precondition that makes it faithful is that a bare term never
    ///   becomes a multi-clause `BooleanQuery` here.** Java's `maxTerms` loop
    ///   zips per *analyzed token* -- `(title:t1 body:t1) (title:t2 body:t2)`
    ///   -- whereas re-parsing a span per field groups per *field*:
    ///   `(title:(t1 t2)) (body:(t1 t2))`. Those are different queries. They
    ///   coincide here only because [`clause_from_analyzed_terms`] turns
    ///   a multi-token bareword into a single [`Clause::Phrase`], never a
    ///   multi-clause `BooleanQuery`, which is Java's `else if (termNum == 0)`
    ///   branch with `maxTerms == 1` -- one sub-query per field, zipped
    ///   trivially. **If `clause_from_analyzed_terms` ever gains a `Boolean`
    ///   shape, this expansion silently produces the wrong query and must be
    ///   changed to zip.** (A single `Analyzer` for all fields is a separate,
    ///   smaller point: Java calls `super.getFieldQuery(fields[i], ...)` per
    ///   field so a *per-field* analyzer can differ, and this parser has
    ///   none, so every field's parse of the same span is identical.)
    /// - **Nested groups expand once, not twice.** `used_default_field` is
    ///   cleared here, so `(cat dog)` has each of `cat`/`dog` expanded by its
    ///   own `parse_boosted_atom` and the enclosing group sees no bare field
    ///   left to expand. The re-parse also runs with `multi_fields` disabled
    ///   for the same reason.
    ///
    /// A per-field boost wraps that field's clause in a [`BoostQuery`], as
    /// Java's `new BoostQuery(q, boost)` does. A `^n` written in the query
    /// text is *inside* each expansion here and *outside* the disjunction in
    /// Java; the two score identically, because a `SHOULD` `BooleanQuery`
    /// sums its clauses' scores and `sum(b * s_i) == b * sum(s_i)`.
    fn expand_across_fields(
        &mut self,
        atom_start: usize,
        first: Clause,
    ) -> Result<Clause, ParseError> {
        let atom_end = self.pos;
        let fields = self.multi_fields;
        let mut should = Vec::with_capacity(fields.len());
        should.push(apply_field_boost(first, fields[0].1));
        for (field, boost) in &fields[1..] {
            let mut sub = Parser {
                chars: self.chars,
                pos: atom_start,
                default_field: Some(field),
                analyzer: self.analyzer,
                default_operator: self.default_operator,
                multi_fields: &[],
                used_default_field: false,
            };
            let clause = sub.parse_atom_with_suffixes()?;
            debug_assert_eq!(
                sub.pos, atom_end,
                "re-parsing the same span must consume the same characters"
            );
            should.push(apply_field_boost(clause, *boost));
        }
        self.pos = atom_end;
        Ok(Clause::Boolean(Box::new(
            BooleanQuery::new().with_should(should),
        )))
    }

    /// The body of what used to be `parse_boosted_atom`: one atom plus its
    /// `^boost` / `~slop` suffixes, in either order.
    fn parse_atom_with_suffixes(&mut self) -> Result<Clause, ParseError> {
        let atom = self.parse_atom()?;
        let (mut clause, bare) = atom;

        let mut boost: Option<f32> = None;
        if self.peek() == Some('^') {
            boost = Some(self.parse_boost()?);
            if self.peek() == Some('~') {
                clause = self.parse_tilde_suffix(clause, bare)?;
            }
        } else if self.peek() == Some('~') {
            clause = self.parse_tilde_suffix(clause, bare)?;
            if self.peek() == Some('^') {
                boost = Some(self.parse_boost()?);
            }
        }

        if let Some(boost) = boost {
            return Ok(Clause::Boost(Box::new(BoostQuery::new(clause, boost))));
        }
        Ok(clause)
    }

    /// `<CARAT> <NUMBER>`.
    fn parse_boost(&mut self) -> Result<f32, ParseError> {
        let start = self.pos;
        self.advance(); // consume '^'
        let num_start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.') {
            self.pos += 1;
        }
        let text: String = self.chars[num_start..self.pos].iter().collect();
        let boost: f32 = text
            .parse()
            .map_err(|_| ParseError::InvalidBoost(start, text.clone()))?;
        if !boost.is_finite() {
            return Err(ParseError::InvalidBoost(start, text));
        }
        Ok(boost)
    }

    /// `<FUZZY_SLOP>` (`'~' [digits ['.' digits]]`) applied to an already-built
    /// atom -- real Lucene's one token serving two purposes:
    ///
    /// - after a quoted phrase it is the **phrase slop**
    ///   (`handleQuotedTerm`: `(int) Float.parseFloat(...)`, so a fractional
    ///   value truncates, and a bare `~` leaves the default slop of 0);
    /// - after a bareword it is the **fuzzy edit distance**
    ///   (`handleBareFuzzy`), rejected when fractional and `>= 1.0`
    ///   ("Fractional edit distances are not allowed!") and capped at
    ///   `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE == 2` here;
    /// - after a wildcard/prefix bareword it is **ignored**, because
    ///   `handleBareTokenQuery` checks `wildcard`/`prefix` *before* `fuzzy`;
    /// - after anything else (a group, a range) real Lucene's grammar has no
    ///   production for it at all, so it is a
    ///   [`ParseError::UnsupportedSyntax`] rather than a silent no-op.
    fn parse_tilde_suffix(
        &mut self,
        clause: Clause,
        bare: Option<BareToken>,
    ) -> Result<Clause, ParseError> {
        let tilde_pos = self.pos;
        self.advance(); // consume '~'
        let digits_start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        let integer_part: String = self.chars[digits_start..self.pos].iter().collect();
        let mut fractional = false;
        if self.peek() == Some('.') {
            fractional = true;
            self.pos += 1;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
        }
        let text: String = self.chars[digits_start..self.pos].iter().collect();

        match clause {
            Clause::Phrase(phrase) => {
                // `handleQuotedTerm`: `(int) Float.parseFloat(image)`, i.e. a
                // fractional slop truncates toward zero; an unparsable value
                // silently keeps the default (Java swallows the exception).
                let slop = text.parse::<f32>().map(|v| v as u32).unwrap_or(0);
                Ok(Clause::Phrase(phrase.with_slop(slop)))
            }
            // `handleBareTokenQuery` checks wildcard/prefix before fuzzy.
            other @ (Clause::Wildcard(_) | Clause::Prefix(_)) => Ok(other),
            _ => {
                let Some(BareToken::Text { field, text: term }) = bare else {
                    return Err(ParseError::UnsupportedSyntax(
                        tilde_pos,
                        "'~' is only meaningful after a quoted phrase or a bareword".to_string(),
                    ));
                };
                if fractional {
                    return Err(ParseError::InvalidFuzziness(tilde_pos, text));
                }
                let mut fuzzy = FuzzyQuery::new(field, term);
                if !integer_part.is_empty() {
                    let edits: u32 = integer_part.parse().map_err(|_| {
                        ParseError::InvalidFuzziness(tilde_pos, integer_part.clone())
                    })?;
                    if edits > 2 {
                        return Err(ParseError::InvalidFuzziness(tilde_pos, integer_part));
                    }
                    fuzzy = fuzzy.with_max_edits(edits as u8);
                }
                Ok(Clause::Fuzzy(fuzzy))
            }
        }
    }

    /// One atom, plus (when it was a plain bareword) the raw token it came
    /// from -- [`Self::parse_tilde_suffix`] needs the *unanalyzed* term text
    /// to build a `FuzzyQuery`, since real Lucene's `getFuzzyQuery` never runs
    /// the analyzer over a fuzzy term.
    fn parse_atom(&mut self) -> Result<(Clause, Option<BareToken>), ParseError> {
        match self.peek() {
            None => Err(ParseError::UnexpectedEnd("an atom")),
            Some('(') => Ok((self.parse_group()?, None)),
            Some('[') | Some('{') => {
                let start = self.pos;
                let field = self
                    .default_field
                    .map(str::to_string)
                    .ok_or(ParseError::MissingField(start))?;
                self.used_default_field = true;
                Ok((self.parse_range(&field)?, None))
            }
            Some(')') => Err(ParseError::UnexpectedChar(self.pos, ')')),
            _ => self.parse_term(),
        }
    }

    fn parse_group(&mut self) -> Result<Clause, ParseError> {
        let open_pos = self.pos;
        self.advance(); // consume '('
        self.skip_ws();
        if self.peek() == Some(')') {
            self.advance();
            return Err(ParseError::UnexpectedChar(open_pos, '('));
        }
        let inner = self.parse_clause_list(true)?;
        self.skip_ws();
        if self.peek() != Some(')') {
            return Err(ParseError::UnclosedParen(open_pos));
        }
        self.advance(); // consume ')'
        Ok(inner)
    }

    /// `(field ':')? termbody`
    fn parse_term(&mut self) -> Result<(Clause, Option<BareToken>), ParseError> {
        let start = self.pos;
        let field = self.try_parse_field()?;
        let field = match field {
            Some(f) => f,
            None => {
                self.used_default_field = true;
                self.default_field
                    .map(str::to_string)
                    .ok_or(ParseError::MissingField(start))?
            }
        };

        match self.peek() {
            Some('"') => Ok((self.parse_phrase(&field)?, None)),
            Some('/') => Ok((self.parse_regexp(&field)?, None)),
            Some('[') | Some('{') => Ok((self.parse_range(&field)?, None)),
            None => Err(ParseError::UnexpectedEnd("a term after ':'")),
            _ => self.parse_wordterm(&field),
        }
    }

    /// `('[' | '{') bound 'TO' bound (']' | '}')` -- a numeric range, called
    /// with `self.pos` at the opening delimiter. `bound` is `*` (open end) or
    /// an optionally-negative decimal `i64`; see the module doc comment for
    /// the exact supported/deferred syntax.
    ///
    /// Each side's delimiter is inclusive/exclusive independently, matching
    /// real classic `QueryParser`: `[` / `]` mean inclusive, `{` / `}` mean
    /// exclusive, and the two sides may mix (e.g. `{a TO b]`). Since
    /// [`PointsRangeQuery`] only stores an inclusive `[min, max]` `i64` pair
    /// (there is no separate min/max-inclusive flag -- see that struct's doc
    /// comment), an exclusive *literal* bound is turned into the equivalent
    /// inclusive one here by stepping it one closer to the other bound
    /// (`min + 1` / `max - 1`, saturating so an exclusive bound already at
    /// `i64::MAX`/`i64::MIN` doesn't overflow). This mirrors real Lucene's own
    /// `QueryParserBase.getRangeQuery`, which does the identical adjustment
    /// before constructing the (inclusive-only) `PointRangeQuery`. An
    /// exclusive `*` (open) bound is left untouched -- `*` denotes "no bound
    /// at all" rather than a literal value to step away from, so
    /// `{* TO 100]`'s low end is still fully unbounded, same as `[* TO 100]`.
    fn parse_range(&mut self, field: &str) -> Result<Clause, ParseError> {
        let open_pos = self.pos;
        let min_inclusive = self.peek() == Some('[');
        self.advance(); // consume '[' or '{'
        self.skip_ws();
        let (min, min_is_open) = self.parse_range_bound(i64::MIN, open_pos)?;
        self.skip_ws();
        self.expect_keyword("TO", open_pos)?;
        self.skip_ws();
        let (max, max_is_open) = self.parse_range_bound(i64::MAX, open_pos)?;
        self.skip_ws();
        let max_inclusive = match self.peek() {
            Some(']') => true,
            Some('}') => false,
            _ => {
                return Err(ParseError::InvalidRangeBound(
                    open_pos,
                    "expected closing ']' or '}'".to_string(),
                ));
            }
        };
        self.advance(); // consume ']' or '}'

        let min = if !min_inclusive && !min_is_open {
            min.saturating_add(1)
        } else {
            min
        };
        let max = if !max_inclusive && !max_is_open {
            max.saturating_sub(1)
        } else {
            max
        };
        Ok(Clause::PointsRange(PointsRangeQuery::new(field, min, max)))
    }

    /// One `[`/`{`-range bound: `*` (mapped to `open_value`, second element
    /// `true`) or a plain, optionally-negative, decimal `i64` (second element
    /// `false`) -- the second element tells [`Self::parse_range`] whether the
    /// bound was a literal value (eligible for the exclusive +-1/-1 step) or
    /// an open `*` (never stepped).
    fn parse_range_bound(
        &mut self,
        open_value: i64,
        open_pos: usize,
    ) -> Result<(i64, bool), ParseError> {
        let bound_start = self.pos;
        if self.peek() == Some('*') {
            self.advance();
            return Ok((open_value, true));
        }
        if self.peek() == Some('-') {
            self.advance();
        }
        let digits_start = self.pos;
        while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
            self.pos += 1;
        }
        if self.pos == digits_start {
            let text: String = self.chars[bound_start..self.pos].iter().collect();
            return Err(ParseError::InvalidRangeBound(
                open_pos,
                format!("expected a number or '*', found {text:?}"),
            ));
        }
        let text: String = self.chars[bound_start..self.pos].iter().collect();
        let value = text.parse::<i64>().map_err(|_| {
            ParseError::InvalidRangeBound(open_pos, format!("invalid integer {text:?}"))
        })?;
        Ok((value, false))
    }

    /// Consumes exactly the literal `keyword` (case-sensitive), preceded and
    /// followed by nothing (caller handles surrounding whitespace) --
    /// requires a word boundary after it (not immediately followed by
    /// another identifier char) so `TOxyz` isn't mistaken for `TO`.
    fn expect_keyword(&mut self, keyword: &str, open_pos: usize) -> Result<(), ParseError> {
        let start = self.pos;
        for expected in keyword.chars() {
            if self.peek() != Some(expected) {
                let found: String = self.chars[start..self.pos.min(self.chars.len())]
                    .iter()
                    .collect();
                return Err(ParseError::InvalidRangeBound(
                    open_pos,
                    format!("expected {keyword:?}, found {found:?}"),
                ));
            }
            self.advance();
        }
        if matches!(self.peek(), Some(c) if !c.is_whitespace()) {
            return Err(ParseError::InvalidRangeBound(
                open_pos,
                format!("expected {keyword:?} followed by whitespace"),
            ));
        }
        Ok(())
    }

    /// Looks ahead for `identifier ':'` and, if found, consumes it and
    /// returns the field name; otherwise consumes nothing and returns
    /// `None`.
    fn try_parse_field(&mut self) -> Result<Option<String>, ParseError> {
        let start = self.pos;
        let mut i = self.pos;
        while i < self.chars.len() && is_field_char(self.chars[i]) {
            i += 1;
        }
        if i > start && self.chars.get(i) == Some(&':') {
            let name: String = self.chars[start..i].iter().collect();
            self.pos = i + 1;
            return Ok(Some(name));
        }
        Ok(None)
    }

    fn parse_phrase(&mut self, field: &str) -> Result<Clause, ParseError> {
        let open_pos = self.pos;
        self.advance(); // consume opening '"'
        let mut text = String::new();
        loop {
            match self.advance() {
                None => return Err(ParseError::UnclosedQuote(open_pos)),
                Some('"') => break,
                Some('\\') => match self.advance() {
                    None => return Err(ParseError::UnclosedQuote(open_pos)),
                    Some(c) => text.push(c),
                },
                Some(c) => text.push(c),
            }
        }
        // Real Lucene's `QueryParser` analyzes phrase query text word-by-word
        // too (not the whole phrase as one blob -- that would let the
        // tokenizer merge words across the original whitespace boundaries).
        // Each whitespace-separated word gets the same zero/one/multi-token
        // treatment as a bareword, but spliced flat into the phrase's term
        // sequence in order (a "multi-token per word" here just means that
        // one input word can contribute more than one phrase position, e.g.
        // an analyzer that splits "state-of-the-art" into several tokens);
        // a word that analyzes to zero tokens (a stopword) simply
        // contributes nothing, same as real `StopFilter` removing it from a
        // phrase's token stream.
        let terms: Vec<String> = text
            .split_whitespace()
            .flat_map(|word| analyze_term_text(self.analyzer, word))
            .collect();
        if terms.is_empty() {
            return Ok(no_match_clause());
        }
        Ok(Clause::Phrase(PhraseQuery::new(field, terms)))
    }

    fn parse_regexp(&mut self, field: &str) -> Result<Clause, ParseError> {
        let open_pos = self.pos;
        self.advance(); // consume opening '/'
        let mut text = String::new();
        loop {
            match self.advance() {
                None => return Err(ParseError::UnclosedRegexp(open_pos)),
                Some('/') => break,
                Some('\\') => match self.advance() {
                    None => return Err(ParseError::UnclosedRegexp(open_pos)),
                    Some('/') => text.push('/'),
                    Some(c) => {
                        text.push('\\');
                        text.push(c);
                    }
                },
                Some(c) => text.push(c),
            }
        }
        Ok(Clause::Regexp(RegexpQuery::new(field, text)))
    }

    /// A bareword: runs of non-whitespace, non-`"/():^~` characters (with
    /// `\`-escaping of any byte). Decides between `Term`/`Wildcard`/`Prefix`
    /// per the module doc's disambiguation rules; a trailing `~` (fuzzy) is
    /// [`Self::parse_tilde_suffix`]'s job, which is why a plain bareword also
    /// returns its raw, *unanalyzed* text as a [`BareToken`].
    fn parse_wordterm(&mut self, field: &str) -> Result<(Clause, Option<BareToken>), ParseError> {
        // `text` is the fully-unescaped bareword, used for Term/Fuzzy/Prefix
        // (none of which re-interpret `\` at resolve time -- `PrefixQuery`
        // never does, and Term/Fuzzy match byte-for-byte). `wildcard_text`
        // instead preserves the backslash in front of an escaped `*`/`?`/`\`
        // (re-escaping it as `\\X`), since `WildcardPattern::new` (the
        // consumer for `Clause::Wildcard`) does its OWN `\`-escape parsing
        // at resolve time -- if this parser stripped the backslash here,
        // an escaped literal `*` mixed with a genuine unescaped `*` elsewhere
        // in the same term would become indistinguishable from a real
        // wildcard operator once handed to `WildcardPattern::new`, silently
        // turning an intended literal into a live wildcard match.
        let mut text = String::new();
        let mut wildcard_text = String::new();
        let mut has_wildcard_char = false;
        // Counts/positions of GENUINE (unescaped) wildcard operators only --
        // derived here, during the scan, rather than by re-inspecting `text`
        // afterward, since `text` has already lost the distinction between an
        // escaped literal `*`/`?` and a real one.
        let mut star_count = 0usize;
        let mut has_question = false;
        let mut last_char_is_unescaped_star = false;

        loop {
            match self.peek() {
                None => break,
                Some(c) if is_term_stop_char(c) => break,
                Some('\\') => {
                    self.advance();
                    match self.advance() {
                        None => return Err(ParseError::UnexpectedEnd("a character after '\\'")),
                        Some(c) => {
                            text.push(c);
                            wildcard_text.push('\\');
                            wildcard_text.push(c);
                            last_char_is_unescaped_star = false;
                        }
                    }
                }
                Some(c) => {
                    self.advance();
                    if c == '*' || c == '?' {
                        has_wildcard_char = true;
                    }
                    if c == '*' {
                        star_count += 1;
                    }
                    if c == '?' {
                        has_question = true;
                    }
                    last_char_is_unescaped_star = c == '*';
                    text.push(c);
                    wildcard_text.push(c);
                }
            }
        }

        if text.is_empty() {
            return Err(ParseError::UnexpectedChar(
                self.pos,
                self.peek().unwrap_or(' '),
            ));
        }

        if has_wildcard_char {
            // A prefix query is exactly "one genuine, unescaped trailing
            // star, no genuine `?` anywhere" -- checked against the counts
            // gathered above (real operators only), not by re-scanning
            // `text` (which can no longer tell an escaped `*` from a real
            // one). An escaped `*` elsewhere in the term correctly falls
            // through to the `Wildcard` branch below instead, since
            // `star_count`/`has_question` never counted it.
            let only_trailing_star =
                star_count == 1 && last_char_is_unescaped_star && !has_question;
            if only_trailing_star {
                // Safe to strip the last (unescaped, real) `*` off `text`
                // for the prefix literal: PrefixQuery never re-interprets
                // backslashes, so the fully-unescaped `text` is correct here.
                let prefix = text[..text.len() - 1].to_string();
                return Ok((Clause::Prefix(PrefixQuery::new(field, prefix)), None));
            }
            return Ok((
                Clause::Wildcard(WildcardQuery::new(field, wildcard_text)),
                None,
            ));
        }

        let terms = analyze_term_text(self.analyzer, &text);
        Ok((
            clause_from_analyzed_terms(field, terms),
            Some(BareToken::Text {
                field: field.to_string(),
                text,
            }),
        ))
    }
}

/// The raw bareword one atom was built from, carried alongside the built
/// [`Clause`] so a following `~` can build a `FuzzyQuery` from the
/// *unanalyzed* term -- real Lucene's `getFuzzyQuery` never runs the analyzer
/// over a fuzzy term either.
enum BareToken {
    Text { field: String, text: String },
}

fn is_field_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_' || c == '-' || c == '.'
}

/// Characters that end a bareword token -- whitespace, the delimiters this
/// grammar gives special meaning to, and `^`/`~` (handled by their own
/// lookahead in [`Parser::parse_boosted_atom`]/[`Parser::parse_wordterm`]).
fn is_term_stop_char(c: char) -> bool {
    c.is_whitespace() || matches!(c, '"' | '/' | '(' | ')' | ':' | '^' | '~')
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{BooleanQuery, PrefixQuery, WildcardQuery};

    // --- `AND`/`OR`/`NOT` operators (`QueryParserBase.addClause`) ---

    fn boolean(clause: &Clause) -> &BooleanQuery {
        match clause {
            Clause::Boolean(b) => b,
            other => panic!("expected a BooleanQuery, got {other:?}"),
        }
    }

    // --- `MultiFieldQueryParser` -------------------------------------------

    fn term(field: &str, t: &str) -> Clause {
        Clause::Term(TermQuery::new(field, t))
    }

    fn should_over(clauses: Vec<Clause>) -> Clause {
        Clause::Boolean(Box::new(BooleanQuery::new().with_should(clauses)))
    }

    #[test]
    fn a_bare_term_fans_out_across_every_field_as_should() {
        // `getMultiFieldQuery` adds each field's sub-query with Occur.SHOULD.
        assert_eq!(
            parse_multi_field_query("cat", &["title", "body"], None).unwrap(),
            should_over(vec![term("title", "cat"), term("body", "cat")])
        );
    }

    #[test]
    fn an_explicitly_fielded_term_is_left_alone() {
        // Java's `field != null` branch: no fan-out, no boost lookup.
        let clause = parse_multi_field_query("title:cat dog", &["title", "body"], None).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.should,
            vec![
                term("title", "cat"),
                should_over(vec![term("title", "dog"), term("body", "dog")]),
            ]
        );
    }

    /// The reason the disjunction has to sit at the *leaf*: with
    /// `AND_OPERATOR`, Java documents `+(title:t1 body:t1) +(title:t2 body:t2)`
    /// -- each term required in *some* field. Expanding at the top instead
    /// (`+(title:t1 title:t2) +(body:t1 body:t2)`) would require both terms in
    /// the same field, a different query.
    #[test]
    fn the_disjunction_sits_under_the_conjunction_not_over_it() {
        let clause = parse_multi_field_query_with_boosts(
            "cat dog",
            &[("title", 1.0), ("body", 1.0)],
            None,
            DefaultOperator::And,
        )
        .unwrap();
        let b = boolean(&clause);
        assert!(b.should.is_empty());
        assert_eq!(
            b.must,
            vec![
                should_over(vec![term("title", "cat"), term("body", "cat")]),
                should_over(vec![term("title", "dog"), term("body", "dog")]),
            ]
        );
    }

    #[test]
    fn per_field_boosts_wrap_each_fields_own_clause() {
        // Java's documented example: `+(title:t^5.0 body:t^10.0)`.
        let clause = parse_multi_field_query_with_boosts(
            "cat",
            &[("title", 5.0), ("body", 10.0)],
            None,
            DefaultOperator::Or,
        )
        .unwrap();
        assert_eq!(
            clause,
            should_over(vec![
                Clause::Boost(Box::new(BoostQuery::new(term("title", "cat"), 5.0))),
                Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 10.0))),
            ])
        );
        // A boost of 1.0 is Java's "no entry in the boosts map": unwrapped.
        assert_eq!(
            parse_multi_field_query_with_boosts(
                "cat",
                &[("title", 1.0), ("body", 2.0)],
                None,
                DefaultOperator::Or
            )
            .unwrap(),
            should_over(vec![
                term("title", "cat"),
                Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 2.0))),
            ])
        );
    }

    #[test]
    fn every_bare_atom_shape_fans_out_not_just_plain_terms() {
        // `getPrefixQuery`/`getWildcardQuery`/`getFuzzyQuery`/`getRegexpQuery`/
        // `getRangeQuery` all have the same `field == null` fan-out.
        let fields = ["title", "body"];
        assert_eq!(
            parse_multi_field_query("ca*", &fields, None).unwrap(),
            should_over(vec![
                Clause::Prefix(PrefixQuery::new("title", "ca")),
                Clause::Prefix(PrefixQuery::new("body", "ca")),
            ])
        );
        let wild = parse_multi_field_query("c?t", &fields, None).unwrap();
        assert_eq!(
            wild,
            should_over(vec![
                Clause::Wildcard(WildcardQuery::new("title", "c?t")),
                Clause::Wildcard(WildcardQuery::new("body", "c?t")),
            ])
        );
        // A quoted phrase, and a phrase with slop.
        let phrase = parse_multi_field_query("\"a b\"~2", &fields, None).unwrap();
        let b = boolean(&phrase);
        assert_eq!(b.should.len(), 2);
        // And a range, whose bare form uses the default field in Java too.
        let range = parse_multi_field_query("[1 TO 9]", &fields, None).unwrap();
        assert_eq!(boolean(&range).should.len(), 2);
    }

    #[test]
    fn a_group_expands_its_members_once_each_not_the_whole_group_again() {
        // `(cat dog)` must be `((title:cat body:cat) (title:dog body:dog))`,
        // not a group duplicated per field.
        let clause = parse_multi_field_query("(cat dog)", &["title", "body"], None).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.should,
            vec![
                should_over(vec![term("title", "cat"), term("body", "cat")]),
                should_over(vec![term("title", "dog"), term("body", "dog")]),
            ]
        );
    }

    #[test]
    fn a_query_text_boost_over_a_fanned_out_atom_keeps_working() {
        // `cat^2` over two fields: the boost lands inside each expansion here
        // and outside the disjunction in Java. A SHOULD BooleanQuery sums its
        // clauses' scores, so `sum(b * s_i) == b * sum(s_i)` -- the same query.
        let clause = parse_multi_field_query("cat^2", &["title", "body"], None).unwrap();
        assert_eq!(
            clause,
            should_over(vec![
                Clause::Boost(Box::new(BoostQuery::new(term("title", "cat"), 2.0))),
                Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 2.0))),
            ])
        );
    }

    #[test]
    fn one_field_is_exactly_the_single_field_parser() {
        assert_eq!(
            parse_multi_field_query("cat AND dog", &["body"], None).unwrap(),
            parse_query("cat AND dog", Some("body")).unwrap()
        );
    }

    #[test]
    fn no_fields_and_empty_input_are_clean_errors() {
        assert!(matches!(
            parse_multi_field_query("cat", &[], None),
            Err(ParseError::MissingField(0))
        ));
        assert!(matches!(
            parse_multi_field_query("   ", &["body"], None),
            Err(ParseError::EmptyQuery)
        ));
    }

    #[test]
    fn a_parse_error_inside_a_fanned_out_atom_still_surfaces() {
        // The re-parse must propagate its error, not swallow it.
        assert!(parse_multi_field_query("cat^bad", &["title", "body"], None).is_err());
    }

    #[test]
    fn multi_field_parsing_runs_the_analyzer_the_same_way_for_every_field() {
        let analyzer = lucene_analysis::Analyzer::standard(None);
        let clause = parse_multi_field_query("Cats", &["title", "body"], Some(&analyzer)).unwrap();
        let b = boolean(&clause);
        assert_eq!(b.should.len(), 2);
        // Whatever the analyzer produced, both fields got the same term.
        let terms: Vec<String> = b
            .should
            .iter()
            .map(|c| match c {
                Clause::Term(t) => String::from_utf8(t.term.clone()).unwrap(),
                other => panic!("expected a term, got {other:?}"),
            })
            .collect();
        assert_eq!(terms[0], terms[1]);
    }

    #[test]
    fn and_makes_both_sides_required() {
        // Java's retroactive rewrite: the clause *before* an AND becomes MUST
        // too, so `a AND b` is `+a +b`, not `a +b`.
        let clause = parse_query("cat AND dog", Some("body")).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.must,
            vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ]
        );
        assert!(b.should.is_empty() && b.must_not.is_empty());
    }

    #[test]
    fn ampersand_ampersand_is_an_alias_for_and() {
        assert_eq!(
            parse_query("cat && dog", Some("body")).unwrap(),
            parse_query("cat AND dog", Some("body")).unwrap()
        );
        assert_eq!(
            parse_query("cat || dog", Some("body")).unwrap(),
            parse_query("cat OR dog", Some("body")).unwrap()
        );
        assert_eq!(
            parse_query("cat AND !dog", Some("body")).unwrap(),
            parse_query("cat AND NOT dog", Some("body")).unwrap()
        );
    }

    #[test]
    fn or_under_the_default_or_operator_leaves_both_sides_optional() {
        let clause = parse_query("cat OR dog", Some("body")).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.should,
            vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ]
        );
        assert!(b.must.is_empty());
        // Adjacent clauses with no conjunction mean the same thing under OR.
        assert_eq!(clause, parse_query("cat dog", Some("body")).unwrap());
    }

    #[test]
    fn not_is_a_modifier_identical_to_minus() {
        let with_not = parse_query("cat AND NOT dog", Some("body")).unwrap();
        let with_minus = parse_query("cat AND -dog", Some("body")).unwrap();
        assert_eq!(with_not, with_minus);
        let b = boolean(&with_not);
        assert_eq!(b.must, vec![Clause::Term(TermQuery::new("body", "cat"))]);
        assert_eq!(
            b.must_not,
            vec![Clause::Term(TermQuery::new("body", "dog"))]
        );
    }

    #[test]
    fn and_does_not_promote_a_prohibited_preceding_clause() {
        // "unless it's already prohibited": `-a AND b` keeps `a` as MUST_NOT.
        let clause = parse_query("-cat AND dog", Some("body")).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.must_not,
            vec![Clause::Term(TermQuery::new("body", "cat"))]
        );
        assert_eq!(b.must, vec![Clause::Term(TermQuery::new("body", "dog"))]);
        assert!(b.should.is_empty());
    }

    #[test]
    fn default_and_operator_makes_adjacent_clauses_required() {
        let clause =
            parse_query_with_operator("cat dog", Some("body"), None, DefaultOperator::And).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.must,
            vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ]
        );
        assert!(b.should.is_empty());
    }

    #[test]
    fn default_and_operator_or_demotes_the_preceding_clause() {
        // The second retroactive rewrite: under AND_OPERATOR, `a OR b` must
        // come out `a b`, not `+a b`.
        let clause =
            parse_query_with_operator("cat OR dog", Some("body"), None, DefaultOperator::And)
                .unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.should,
            vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ]
        );
        assert!(b.must.is_empty());
    }

    #[test]
    fn default_and_operator_keeps_an_explicit_plus_required() {
        let clause =
            parse_query_with_operator("+cat OR dog", Some("body"), None, DefaultOperator::And)
                .unwrap();
        let b = boolean(&clause);
        // `+cat` is demoted by the following OR exactly as a bare `cat` would
        // be -- Java's rewrite only skips *prohibited* clauses.
        assert_eq!(b.should.len(), 2);
    }

    #[test]
    fn and_or_not_are_only_operators_as_whole_uppercase_tokens() {
        // "ANDROID"/"NOTHING" are longer tokens, "and" is lowercase: all
        // ordinary terms, exactly as JavaCC's tokenizer decides.
        assert_eq!(
            parse_query("ANDROID", Some("body")).unwrap(),
            Clause::Term(TermQuery::new("body", "ANDROID"))
        );
        assert_eq!(
            parse_query("NOTHING", Some("body")).unwrap(),
            Clause::Term(TermQuery::new("body", "NOTHING"))
        );
        let lowercase = parse_query("cat and dog", Some("body")).unwrap();
        let b = boolean(&lowercase);
        assert_eq!(b.should.len(), 3, "lowercase \"and\" is a term");
        assert!(b.must.is_empty());
    }

    #[test]
    fn operators_apply_inside_a_group_independently() {
        let clause = parse_query("(cat AND dog) OR bird", Some("body")).unwrap();
        let b = boolean(&clause);
        assert_eq!(b.should.len(), 2);
        let inner = boolean(&b.should[0]);
        assert_eq!(inner.must.len(), 2);
    }

    #[test]
    fn a_trailing_conjunction_is_an_error_not_a_silent_drop() {
        assert_eq!(
            parse_query("cat AND", Some("body")).unwrap_err(),
            ParseError::UnexpectedEnd("a clause after a conjunction")
        );
    }

    #[test]
    fn a_leading_conjunction_is_parsed_as_a_term_like_javas_grammar() {
        // Java's `Query()` production only allows a conjunction from the
        // second clause on, so a leading `AND` is not a conjunction at all;
        // here it falls through to `Modifiers()`/`Clause()` and lexes as the
        // bareword it is, giving two SHOULD clauses.
        let clause = parse_query("AND cat", Some("body")).unwrap();
        let b = boolean(&clause);
        assert_eq!(
            b.should,
            vec![
                Clause::Term(TermQuery::new("body", "AND")),
                Clause::Term(TermQuery::new("body", "cat")),
            ]
        );
    }

    // --- Phrase slop ---

    #[test]
    fn phrase_slop_sets_the_phrase_querys_slop() {
        let clause = parse_query(r#"body:"quick fox"~3"#, None).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(3))
        );
    }

    #[test]
    fn bare_tilde_after_a_phrase_leaves_the_default_slop() {
        let clause = parse_query(r#"body:"quick fox"~"#, None).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
    }

    #[test]
    fn fractional_phrase_slop_truncates_like_javas_int_cast() {
        let clause = parse_query(r#"body:"quick fox"~1.9"#, None).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(1))
        );
    }

    #[test]
    fn phrase_slop_and_boost_parse_in_either_order() {
        let slop_first = parse_query(r#"body:"quick fox"~2^3"#, None).unwrap();
        let boost_first = parse_query(r#"body:"quick fox"^3~2"#, None).unwrap();
        assert_eq!(slop_first, boost_first);
        assert_eq!(
            slop_first,
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(2)),
                3.0,
            )))
        );
    }

    #[test]
    fn fuzzy_and_boost_parse_in_either_order() {
        let fuzzy_first = parse_query("body:cat~1^2", None).unwrap();
        let boost_first = parse_query("body:cat^2~1", None).unwrap();
        assert_eq!(fuzzy_first, boost_first);
        assert_eq!(
            fuzzy_first,
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1)),
                2.0,
            )))
        );
    }

    #[test]
    fn tilde_after_a_wildcard_or_prefix_is_ignored_like_javas_precedence() {
        // `handleBareTokenQuery` checks wildcard/prefix before fuzzy.
        assert_eq!(
            parse_query("body:ca*~2", None).unwrap(),
            Clause::Prefix(PrefixQuery::new("body", "ca"))
        );
        assert_eq!(
            parse_query("body:c*t~2", None).unwrap(),
            Clause::Wildcard(WildcardQuery::new("body", "c*t"))
        );
    }

    #[test]
    fn tilde_after_a_group_is_unsupported_syntax_not_silently_dropped() {
        assert!(matches!(
            parse_query("(cat dog)~2", Some("body")).unwrap_err(),
            ParseError::UnsupportedSyntax(_, _)
        ));
    }

    #[test]
    fn single_bare_term_uses_default_field() {
        let clause = parse_query("cat", Some("body")).unwrap();
        assert_eq!(clause, Clause::Term(TermQuery::new("body", "cat")));
    }

    #[test]
    fn bare_term_with_no_default_field_is_an_error() {
        let err = parse_query("cat", None).unwrap_err();
        assert_eq!(err, ParseError::MissingField(0));
    }

    #[test]
    fn field_prefixed_term() {
        let clause = parse_query("title:cat", Some("body")).unwrap();
        assert_eq!(clause, Clause::Term(TermQuery::new("title", "cat")));
    }

    #[test]
    fn quoted_phrase() {
        let clause = parse_query(r#"body:"quick fox""#, None).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
    }

    #[test]
    fn quoted_phrase_uses_default_field() {
        let clause = parse_query(r#""quick fox""#, Some("body")).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
    }

    #[test]
    fn plus_minus_combination() {
        let clause = parse_query("+cat -dog bird", Some("body")).unwrap();
        assert_eq!(
            clause,
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([TermQuery::new("body", "cat")])
                    .with_should([TermQuery::new("body", "bird")])
                    .with_must_not([TermQuery::new("body", "dog")])
            ))
        );
    }

    #[test]
    fn wildcard_term() {
        let clause = parse_query("body:c?t", None).unwrap();
        assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "c?t")));
    }

    #[test]
    fn wildcard_term_with_interior_star() {
        let clause = parse_query("body:c*t", None).unwrap();
        assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "c*t")));
    }

    #[test]
    fn trailing_star_is_a_prefix_query() {
        let clause = parse_query("body:ca*", None).unwrap();
        assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "ca")));
    }

    #[test]
    fn fuzzy_default_edit_distance() {
        let clause = parse_query("body:cat~", None).unwrap();
        assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "cat")));
    }

    #[test]
    fn fuzzy_explicit_edit_distance() {
        let clause = parse_query("body:cat~1", None).unwrap();
        assert_eq!(
            clause,
            Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1))
        );
    }

    #[test]
    fn fuzzy_edit_distance_over_two_is_an_error() {
        let err = parse_query("body:cat~3", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidFuzziness(_, _)));
    }

    #[test]
    fn fuzzy_fractional_similarity_is_an_error() {
        let err = parse_query("body:cat~0.8", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidFuzziness(_, _)));
    }

    #[test]
    fn regexp_term() {
        let clause = parse_query("body:/ca.*/", None).unwrap();
        assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "ca.*")));
    }

    #[test]
    fn regexp_with_escaped_slash() {
        let clause = parse_query(r"body:/a\/b/", None).unwrap();
        assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "a/b")));
    }

    #[test]
    fn parenthesized_group() {
        let clause = parse_query("(+cat -dog)", Some("body")).unwrap();
        assert_eq!(
            clause,
            Clause::Boolean(Box::new(
                BooleanQuery::new()
                    .with_must([TermQuery::new("body", "cat")])
                    .with_must_not([TermQuery::new("body", "dog")])
            ))
        );
    }

    #[test]
    fn nested_parenthesized_groups() {
        let clause = parse_query("+(cat (dog bird))", Some("body")).unwrap();
        let Clause::Boolean(top) = &clause else {
            panic!("expected Boolean");
        };
        assert_eq!(top.must.len(), 1);
        let Clause::Boolean(inner) = &top.must[0] else {
            panic!("expected nested Boolean");
        };
        assert_eq!(inner.should.len(), 2);
        assert_eq!(inner.should[0], Clause::Term(TermQuery::new("body", "cat")));
        let Clause::Boolean(innermost) = &inner.should[1] else {
            panic!("expected nested Boolean for '(dog bird)'");
        };
        assert_eq!(innermost.should.len(), 2);
    }

    #[test]
    fn boost_suffix_on_term() {
        let clause = parse_query("body:cat^2.5", None).unwrap();
        assert_eq!(
            clause,
            Clause::Boost(Box::new(BoostQuery::new(
                TermQuery::new("body", "cat"),
                2.5
            )))
        );
    }

    #[test]
    fn boost_suffix_on_group() {
        let clause = parse_query("(body:cat body:dog)^3", None).unwrap();
        let Clause::Boost(boost) = &clause else {
            panic!("expected Boost");
        };
        assert_eq!(boost.boost, 3.0);
        assert!(matches!(*boost.inner, Clause::Boolean(_)));
    }

    #[test]
    fn escaped_special_char_is_literal() {
        let clause = parse_query(r"body:ca\*t", None).unwrap();
        assert_eq!(clause, Clause::Term(TermQuery::new("body", "ca*t")));
    }

    /// A term mixing an escaped `*` (must stay literal) with two genuine
    /// wildcard `*`s (so the term is a `Wildcard`, not a `Prefix` -- a
    /// single genuine trailing star alone would take the `Prefix` branch
    /// instead, which is correct there since `PrefixQuery` never
    /// re-interprets escapes) must resolve with the escape still intact:
    /// `WildcardPattern::new` (the resolve-time consumer) must still see the
    /// escape and treat only the genuine `*`s as real operators -- not a
    /// pattern where the escaped `*` and a genuine `*` look identical once
    /// resolved (which would silently turn the intended literal `*` into an
    /// extra live wildcard, over-matching).
    #[test]
    fn escaped_wildcard_char_mixed_with_genuine_ones_stays_literal_at_resolve_time() {
        let clause = parse_query(r"body:a\*b*c*", None).unwrap();
        let Clause::Wildcard(w) = &clause else {
            panic!("expected Wildcard, got {clause:?}");
        };
        // The parser must hand WildcardPattern::new a pattern that still
        // carries the escape (`\*`), not an already-unescaped "a*b*c*" where
        // all three stars are indistinguishable.
        let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
        assert!(
            pattern.matches(b"a*bXYZcXYZ"),
            "the literal `a*b` prefix, anything, then literal `c`, then anything, must match"
        );
        assert!(
            !pattern.matches(b"aXYZbXYZcXYZ"),
            "the escaped `*` must NOT be treated as a live wildcard operator: {pattern:?} vs aXYZbXYZcXYZ"
        );
    }

    /// The same property with the escaped star in the middle instead of
    /// right after the leading literal, closing the same gap from another
    /// position.
    #[test]
    fn escaped_wildcard_char_in_the_middle_stays_literal_at_resolve_time() {
        let clause = parse_query(r"body:a*b\*c*", None).unwrap();
        let Clause::Wildcard(w) = &clause else {
            panic!("expected Wildcard, got {clause:?}");
        };
        let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
        assert!(
            pattern.matches(b"aXYZb*cXYZ"),
            "literal `a`, anything, literal `b*c`, anything, must match"
        );
        assert!(
            !pattern.matches(b"aXYZbXYZcXYZ"),
            "the escaped `*` must NOT be treated as a live wildcard operator"
        );
    }

    #[test]
    fn unclosed_quote_is_a_clean_error() {
        let err = parse_query(r#"body:"quick fox"#, None).unwrap_err();
        assert!(matches!(err, ParseError::UnclosedQuote(_)));
    }

    #[test]
    fn unclosed_paren_is_a_clean_error() {
        let err = parse_query("(body:cat", None).unwrap_err();
        assert!(matches!(err, ParseError::UnclosedParen(_)));
    }

    #[test]
    fn unclosed_regexp_is_a_clean_error() {
        let err = parse_query("body:/ca.*", None).unwrap_err();
        assert!(matches!(err, ParseError::UnclosedRegexp(_)));
    }

    #[test]
    fn unmatched_close_paren_is_a_clean_error() {
        let err = parse_query("body:cat)", None).unwrap_err();
        assert!(matches!(err, ParseError::UnmatchedCloseParen(_)));
    }

    #[test]
    fn inclusive_inclusive_range_query() {
        let clause = parse_query("body:[0 TO 100]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
        );
    }

    #[test]
    fn exclusive_exclusive_range_query() {
        // `{0 TO 100}` excludes both 0 and 100, i.e. equivalent to `[1 TO 99]`.
        let clause = parse_query("body:{0 TO 100}", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 99))
        );
    }

    #[test]
    fn inclusive_exclusive_range_query() {
        // `[0 TO 100}` includes 0, excludes 100.
        let clause = parse_query("body:[0 TO 100}", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 99))
        );
    }

    #[test]
    fn exclusive_inclusive_range_query() {
        // `{0 TO 100]` excludes 0, includes 100.
        let clause = parse_query("body:{0 TO 100]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 100))
        );
    }

    #[test]
    fn bare_exclusive_range_query_is_supported() {
        // The bare-atom `{` arm (no `field:` prefix) is a separate code path
        // from `parse_term`'s -- exercise it directly, not just the
        // field-prefixed siblings above.
        let clause = parse_query("{0 TO 100}", Some("body")).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 99))
        );
    }

    #[test]
    fn exclusive_range_with_open_bound_is_not_stepped() {
        // `*` denotes "no bound", so the exclusive side doesn't get stepped
        // even though it's written with `{`/`}`.
        let clause = parse_query("body:{* TO 100}", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", i64::MIN, 99))
        );
        let clause = parse_query("body:{0 TO *}", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, i64::MAX))
        );
    }

    #[test]
    fn exclusive_range_bound_at_extreme_does_not_overflow() {
        // An exclusive literal bound already at i64::MAX/i64::MIN saturates
        // instead of overflowing when stepped.
        let clause = parse_query(&format!("body:{{{} TO {}}}", i64::MIN, i64::MAX), None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new(
                "body",
                i64::MIN + 1,
                i64::MAX - 1
            ))
        );
    }

    #[test]
    fn mismatched_range_delimiter_is_a_clean_error() {
        let err = parse_query("body:[0 TO 100)", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn non_numeric_range_bound_is_a_clean_error() {
        let err = parse_query("body:[a TO b]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn inclusive_numeric_range_query() {
        let clause = parse_query("body:[0 TO 100]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
        );
    }

    #[test]
    fn inclusive_numeric_range_query_with_negative_bounds() {
        let clause = parse_query("body:[-100 TO -1]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", -100, -1))
        );
    }

    #[test]
    fn range_query_with_star_on_low_end() {
        let clause = parse_query("body:[* TO 100]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", i64::MIN, 100))
        );
    }

    #[test]
    fn range_query_with_star_on_high_end() {
        let clause = parse_query("body:[0 TO *]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, i64::MAX))
        );
    }

    #[test]
    fn range_query_with_star_on_both_ends() {
        let clause = parse_query("body:[* TO *]", None).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new(
                "body",
                i64::MIN,
                i64::MAX
            ))
        );
    }

    #[test]
    fn range_query_uses_default_field_when_bare() {
        let clause = parse_query("[0 TO 100]", Some("body")).unwrap();
        assert_eq!(
            clause,
            Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
        );
    }

    #[test]
    fn range_query_missing_to_keyword_is_a_clean_error() {
        let err = parse_query("body:[0 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_lowercase_to_is_not_recognized() {
        let err = parse_query("body:[0 to 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_missing_closing_bracket_is_a_clean_error() {
        let err = parse_query("body:[0 TO 100", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_missing_min_bound_is_a_clean_error() {
        let err = parse_query("body:[ TO 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_missing_max_bound_is_a_clean_error() {
        let err = parse_query("body:[0 TO ]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_non_numeric_max_bound_is_a_clean_error() {
        let err = parse_query("body:[0 TO a]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_bound_overflowing_i64_is_a_clean_error() {
        let err = parse_query("body:[99999999999999999999 TO 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_to_keyword_without_trailing_word_boundary_is_a_clean_error() {
        // "TOxyz" must not be accepted as the "TO" keyword just because it
        // starts with the right two characters.
        let err = parse_query("body:[0 TOxyz 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
    }

    #[test]
    fn range_query_boosted() {
        let clause = parse_query("body:[0 TO 100]^2", None).unwrap();
        assert_eq!(
            clause,
            Clause::Boost(Box::new(BoostQuery::new(
                crate::query::PointsRangeQuery::new("body", 0, 100),
                2.0
            )))
        );
    }

    #[test]
    fn empty_input_is_an_error() {
        let err = parse_query("", Some("body")).unwrap_err();
        assert_eq!(err, ParseError::EmptyQuery);
    }

    #[test]
    fn whitespace_only_input_is_an_error() {
        let err = parse_query("   ", Some("body")).unwrap_err();
        assert_eq!(err, ParseError::EmptyQuery);
    }

    #[test]
    fn invalid_boost_is_a_clean_error() {
        let err = parse_query("body:cat^abc", None).unwrap_err();
        // '^' followed by non-digit/non-'.' parses as an empty number,
        // which fails to parse as f32.
        assert!(matches!(err, ParseError::InvalidBoost(_, _)));
    }

    #[test]
    fn empty_group_is_an_error() {
        let err = parse_query("()", Some("body")).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedChar(_, '(')));
    }

    #[test]
    fn bare_colon_with_no_field_name_falls_back_to_wordterm() {
        // ':' isn't a valid leading field char, so `try_parse_field` finds
        // no identifier before it and `parse_wordterm` is reached instead --
        // but ':' is itself a term-stop char, so this yields an empty
        // bareword, which is a clean error rather than a panic.
        let err = parse_query(":cat", Some("body")).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedChar(_, ':')));
    }

    #[test]
    fn trailing_backslash_at_end_of_input_is_an_error() {
        let err = parse_query(r"body:cat\", None).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEnd(_)));
    }

    #[test]
    fn unclosed_paren_with_no_atom_inside_is_a_clean_error() {
        // '(' followed only by whitespace/EOF: `parse_clause_list` sees
        // `None` immediately and reports `EmptyQuery` (bubbling up through
        // `parse_group`'s `?`) rather than `UnclosedParen` -- an honest, if
        // slightly imprecise, label for "nothing valid was ever found," and
        // still a clean `Err`, not a panic.
        let err = parse_query("(   ", Some("body")).unwrap_err();
        assert_eq!(err, ParseError::EmptyQuery);
    }

    #[test]
    fn boost_value_too_large_for_f32_is_an_error() {
        // All-digit text that overflows `f32::MAX` parses successfully to
        // `f32::INFINITY` rather than failing `str::parse`, so this exercises
        // the separate `is_finite()` check.
        let query = format!("body:cat^{}", "9".repeat(50));
        let err = parse_query(&query, None).unwrap_err();
        assert!(matches!(err, ParseError::InvalidBoost(_, _)));
    }

    #[test]
    fn trailing_modifier_with_nothing_after_is_an_error() {
        let err = parse_query("body:cat +", Some("body")).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEnd(_)));
    }

    #[test]
    fn bare_range_query_syntax_without_field_prefix_and_no_default_field_is_missing_field() {
        let err = parse_query("[0 TO 100]", None).unwrap_err();
        assert!(matches!(err, ParseError::MissingField(_)));
    }

    #[test]
    fn field_colon_with_nothing_after_is_an_error() {
        let err = parse_query("body:", None).unwrap_err();
        assert!(matches!(err, ParseError::UnexpectedEnd(_)));
    }

    #[test]
    fn unclosed_quote_after_trailing_escape_is_a_clean_error() {
        let err = parse_query("body:\"foo\\", None).unwrap_err();
        assert!(matches!(err, ParseError::UnclosedQuote(_)));
    }

    #[test]
    fn unclosed_regexp_after_trailing_escape_is_a_clean_error() {
        let err = parse_query(r"body:/a\", None).unwrap_err();
        assert!(matches!(err, ParseError::UnclosedRegexp(_)));
    }

    #[test]
    fn regexp_escaped_non_slash_char_keeps_the_backslash_literally() {
        // Only `\/` is special-cased to a literal `/`; any other escaped
        // byte is passed through as `\` + that byte unchanged, left for
        // `RegexpPattern::new`'s own escaping rules to interpret later.
        let clause = parse_query(r"body:/a\d/", None).unwrap();
        assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", r"a\d")));
    }

    // --- Analyzer wiring (task #62) ---

    use lucene_analysis::Analyzer;
    use std::collections::HashSet;

    #[test]
    fn none_analyzer_behavior_is_unchanged() {
        // Every existing test above calls `parse_query`, which now delegates
        // to `parse_query_with_analyzer(.., None)` -- this test additionally
        // pins that calling the two spellings directly produces identical
        // results for a representative case.
        let a = parse_query("Quick", Some("body")).unwrap();
        let b = parse_query_with_analyzer("Quick", Some("body"), None).unwrap();
        assert_eq!(a, b);
        assert_eq!(a, Clause::Term(TermQuery::new("body", "Quick")));
    }

    #[test]
    fn bareword_through_lowercase_only_analyzer_is_lowercased() {
        let analyzer = Analyzer::standard(None);
        let clause = parse_query_with_analyzer("Quick", Some("body"), Some(&analyzer)).unwrap();
        assert_eq!(clause, Clause::Term(TermQuery::new("body", "quick")));
    }

    #[test]
    fn bareword_that_is_a_stopword_yields_no_match_not_panic() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let clause = parse_query_with_analyzer("the", Some("body"), Some(&analyzer)).unwrap();
        assert_eq!(
            clause,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    #[test]
    fn bareword_analyzer_splits_into_multiple_tokens_becomes_phrase() {
        // The analysis-crate tokenizer splits on non-alphanumeric boundaries,
        // so a hyphenated bareword like "state-of-the-art" naturally becomes
        // multiple tokens -- exercising the "analyzer produced >1 token from
        // one bareword" path without needing a custom analyzer.
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let clause =
            parse_query_with_analyzer("state-of-the-art", Some("body"), Some(&analyzer)).unwrap();
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["state", "of", "art"]))
        );
    }

    #[test]
    fn wildcard_pattern_text_is_not_analyzed() {
        // Uppercase letters in a wildcard pattern must survive untouched --
        // the analyzer must never see wildcard/prefix/fuzzy/regexp pattern
        // text.
        let analyzer = Analyzer::standard(None);
        let clause = parse_query_with_analyzer("body:C?T", None, Some(&analyzer)).unwrap();
        assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "C?T")));
    }

    #[test]
    fn prefix_pattern_text_is_not_analyzed() {
        let analyzer = Analyzer::standard(None);
        let clause = parse_query_with_analyzer("body:CA*", None, Some(&analyzer)).unwrap();
        assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "CA")));
    }

    #[test]
    fn fuzzy_pattern_text_is_not_analyzed() {
        let analyzer = Analyzer::standard(None);
        let clause = parse_query_with_analyzer("body:CAT~", None, Some(&analyzer)).unwrap();
        assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "CAT")));
    }

    #[test]
    fn regexp_pattern_text_is_not_analyzed() {
        // A would-be-stopword-shaped substring ("the") inside the pattern
        // must survive verbatim, and case must be untouched.
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let clause = parse_query_with_analyzer("body:/THE.*/", None, Some(&analyzer)).unwrap();
        assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "THE.*")));
    }

    #[test]
    fn quoted_phrase_words_are_analyzed_per_word() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let clause =
            parse_query_with_analyzer(r#"body:"The Quick FOX""#, None, Some(&analyzer)).unwrap();
        // "The" is a stopword and drops out entirely; the rest lowercase.
        assert_eq!(
            clause,
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
        );
    }

    #[test]
    fn quoted_phrase_entirely_stopwords_is_no_match() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let clause = parse_query_with_analyzer(r#"body:"the the""#, None, Some(&analyzer)).unwrap();
        assert_eq!(
            clause,
            Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                filter: Vec::new(),
                should: vec![],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }

    /// Execution-level proof, not just a parsed-`Clause` shape check: an
    /// exclusive bound produced by [`parse_query`] actually excludes the
    /// boundary value when the resulting [`PointsRangeQuery`]'s `min`/`max`
    /// are fed through the real BKD points search path
    /// ([`crate::points_query::search_points_range`]). `Clause::PointsRange`
    /// itself has no resolver yet (see that variant's doc comment), so this
    /// drives `search_points_range` directly with the parser's output --
    /// the same thing a future resolver would do -- rather than going
    /// through `resolve_clause_docs`.
    #[test]
    fn exclusive_bound_excludes_boundary_value_end_to_end() {
        use crate::collector::VecCollector;
        use crate::points_query::search_points_range;
        use lucene_codecs::points::{self, WritePointsField};
        use lucene_store::codec_util::ID_LENGTH;

        fn long_bytes(v: i64) -> [u8; 8] {
            ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
        }

        // Doc 0 -> 10, doc 1 -> 20, doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
        let segment_id = [7u8; ID_LENGTH];
        let points: Vec<(i32, Vec<u8>)> = vec![
            (0, long_bytes(10).to_vec()),
            (1, long_bytes(20).to_vec()),
            (2, long_bytes(30).to_vec()),
            (3, long_bytes(40).to_vec()),
            (4, long_bytes(50).to_vec()),
        ];
        let field = WritePointsField {
            field_number: 1,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 8,
            points,
        };
        let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
        let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();

        // `{10 TO 40}` must exclude both boundary docs (values 10 and 40)
        // and match only the strictly-between ones (20, 30).
        let clause = parse_query("body:{10 TO 40}", None).unwrap();
        let Clause::PointsRange(range) = clause else {
            panic!("expected PointsRange, got {clause:?}");
        };
        assert_eq!(range.min, 11);
        assert_eq!(range.max, 39);
        let min = long_bytes(range.min);
        let max = long_bytes(range.max);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert_eq!(
            collector.docs,
            vec![1, 2],
            "must match only values 20 and 30, excluding boundary values 10 and 40"
        );

        // Sanity check: the equivalent inclusive `[10 TO 40]` does include
        // both boundary docs, proving the exclusion above is really coming
        // from `{`/`}`, not from some unrelated bug.
        let clause = parse_query("body:[10 TO 40]", None).unwrap();
        let Clause::PointsRange(range) = clause else {
            panic!("expected PointsRange, got {clause:?}");
        };
        let min = long_bytes(range.min);
        let max = long_bytes(range.max);
        let mut collector = VecCollector::default();
        search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
        assert_eq!(collector.docs, vec![0, 1, 2, 3]);
    }
}
