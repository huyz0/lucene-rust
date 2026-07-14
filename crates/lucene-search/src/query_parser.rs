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
//! query      := clause*
//! clause     := modifier? atom boost?
//! modifier   := '+' | '-'                  (absence => SHOULD)
//! atom       := group | term
//! group      := '(' query ')'
//! term       := (field ':')? termbody
//! field      := identifier (ASCII letters/digits/'_'/'-'/'.', 1+ chars)
//! termbody   := phrase | regexp | wordterm
//! phrase     := '"' char* '"'              => Clause::Phrase
//! regexp     := '/' char* '/'              => Clause::Regexp
//! wordterm   := bareword ('~' digit*)?     => Clause::Fuzzy (if '~' present)
//!             | bareword                  => Clause::Wildcard / Clause::Prefix / Clause::Term
//! boost      := '^' float
//! ```
//!
//! **Boolean-operator style: `+`/`-` prefixes, not `AND`/`OR`/`NOT`.** Real
//! Lucene's classic `QueryParser` actually supports *both* styles
//! simultaneously with configurable default-operator precedence rules --
//! exactly the "half-supported mix" the task description called out to
//! avoid. This module supports **only** the `+`/`-`/bare-is-SHOULD style**:
//! a bare clause is optional (`Occur::SHOULD`), a `+`-prefixed clause is
//! required (`Occur::MUST`), a `-`-prefixed clause is excluded
//! (`Occur::MUST_NOT`). `AND`/`OR`/`NOT` tokens are treated as ordinary bare
//! terms (or field names), never as operators -- this is unambiguous to
//! parse (no precedence table needed: every clause's role is determined by
//! its own leading character, not by what comes between clauses) and maps
//! directly onto [`crate::query::BooleanQuery`]'s existing
//! `must`/`should`/`must_not` buckets with no translation layer.
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
//! [`ParseError::InvalidFuzziness`], not silently clamped.
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
//! an inclusive `i64` range, parsed into [`Clause::PointsRange`]. Either
//! bound may be `*` for an open end (mapped to `i64::MIN`/`i64::MAX`,
//! matching real Lucene's own unbounded-range convention), and either bound
//! may be a negative decimal integer (e.g. `field:[-100 TO 0]`). The `TO`
//! keyword is matched case-sensitively (real classic `QueryParser`'s grammar
//! requires uppercase `TO` too). **Parsing only**: the resulting
//! [`crate::query::PointsRangeQuery`] is not yet resolved against a segment
//! by anything in this crate -- see that struct's doc comment for the exact
//! deferred scope.
//!
//! ## Deliberately deferred (parse error, not silent misinterpretation)
//!
//! - **Exclusive-bound range queries** (`field:{a TO b}`) -- rejected with
//!   [`ParseError::UnsupportedSyntax`] at the `{` character. A mixed
//!   `field:[a TO b}` opens as an inclusive range (the `[`) but then fails
//!   the closing-bracket check with [`ParseError::InvalidRangeBound`] at the
//!   `}` -- still a clean, typed error, just a different variant than the
//!   pure `{...}` case, since the parser has already committed to the
//!   inclusive-range code path by the time it sees the mismatched closer.
//!   Only the fully-inclusive `[a TO b]` shape is actually supported.
//! - **String/date range queries** (`field:[aaa TO zzz]` over a
//!   non-numeric/`TermRangeQuery`-shaped field) -- a `[min TO max]` whose
//!   bounds don't parse as a plain (optionally negative) decimal integer or
//!   `*` is a [`ParseError::InvalidRangeBound`], not a fallback to string
//!   comparison.
//! - **`AND`/`OR`/`NOT` as real operators with precedence rules** -- not
//!   implemented at all (see above); those words parse as ordinary terms.
//! - **Boosting a group's boost multiplying inner boosts / any boost
//!   algebra beyond one flat `^N` per atom** -- a `^` after a `)` applies
//!   exactly the same single [`crate::query::BoostQuery`] wrap a term/phrase
//!   boost gets, nothing fancier.
//! - **Fuzziness `~` with a fractional similarity (e.g. `term~0.8`, the
//!   `StandardQueryParser` float-similarity convention)** -- only bare `~`
//!   or `~` followed by an integer `0..=2` is accepted; a `~` followed by a
//!   decimal point is a [`ParseError::InvalidFuzziness`].
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
    /// "deliberately deferred" list) -- e.g. exclusive (`{`) or mixed range
    /// syntax.
    #[error("unsupported syntax at byte {0}: {1}")]
    UnsupportedSyntax(usize, String),
    /// A `field:[min TO max]` range bound wasn't `*` and didn't parse as a
    /// plain (optionally negative) decimal `i64`, or the `TO` keyword was
    /// missing/misspelled, or the range wasn't closed by a matching `]`.
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
/// this module's `+`/`-`/bare syntax mapped straight onto real `Occur`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Occur {
    Must,
    Should,
    MustNot,
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
    let bytes: Vec<char> = input.chars().collect();
    let mut parser = Parser {
        chars: &bytes,
        pos: 0,
        default_field,
        analyzer,
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

    /// Parses a sequence of `modifier? atom boost?` clauses until end of
    /// input or (when `inside_group`) a closing `)`, combining them into a
    /// single [`Clause`]: if exactly one clause was found with the default
    /// (`Should`) modifier, that clause is returned unwrapped; otherwise
    /// they're grouped into a [`Clause::Boolean`] by `Occur` bucket.
    fn parse_clause_list(&mut self, inside_group: bool) -> Result<Clause, ParseError> {
        let mut must = Vec::new();
        let mut should = Vec::new();
        let mut must_not = Vec::new();
        let mut only_clause: Option<Clause> = None;
        let mut count = 0usize;

        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(')') if inside_group => break,
                Some(')') => return Err(ParseError::UnmatchedCloseParen(self.pos)),
                _ => {}
            }

            let occur = match self.peek() {
                Some('+') => {
                    self.advance();
                    Occur::Must
                }
                Some('-') => {
                    self.advance();
                    Occur::MustNot
                }
                _ => Occur::Should,
            };

            let clause = self.parse_boosted_atom()?;
            count += 1;
            match occur {
                Occur::Must => must.push(clause.clone()),
                Occur::Should => should.push(clause.clone()),
                Occur::MustNot => must_not.push(clause.clone()),
            }
            if count == 1 && occur == Occur::Should {
                only_clause = Some(clause);
            } else {
                only_clause = None;
            }
        }

        if count == 0 {
            return Err(ParseError::EmptyQuery);
        }
        if count == 1 {
            if let Some(clause) = only_clause {
                return Ok(clause);
            }
        }
        Ok(Clause::Boolean(Box::new(BooleanQuery {
            must,
            should,
            must_not,
            minimum_should_match: 0,
        })))
    }

    /// `atom boost?` -- parses one atom (a group or a term) then an optional
    /// trailing `^number`.
    fn parse_boosted_atom(&mut self) -> Result<Clause, ParseError> {
        let clause = self.parse_atom()?;
        self.skip_ws_within_atom();
        if self.peek() == Some('^') {
            let start = self.pos;
            self.advance();
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
            return Ok(Clause::Boost(Box::new(BoostQuery::new(clause, boost))));
        }
        Ok(clause)
    }

    /// No whitespace is actually permitted between an atom and its `^boost`
    /// in this grammar (real Lucene doesn't allow it either) -- this is a
    /// no-op placeholder kept so `parse_boosted_atom`'s intent (look for `^`
    /// immediately after the atom) reads clearly at the call site.
    fn skip_ws_within_atom(&mut self) {}

    fn parse_atom(&mut self) -> Result<Clause, ParseError> {
        match self.peek() {
            None => Err(ParseError::UnexpectedEnd("an atom")),
            Some('(') => self.parse_group(),
            Some('[') => {
                let start = self.pos;
                let field = self
                    .default_field
                    .map(str::to_string)
                    .ok_or(ParseError::MissingField(start))?;
                self.parse_range(&field)
            }
            Some('{') => Err(ParseError::UnsupportedSyntax(
                self.pos,
                "exclusive range queries ({a TO b}) are not supported".to_string(),
            )),
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
    fn parse_term(&mut self) -> Result<Clause, ParseError> {
        let start = self.pos;
        let field = self.try_parse_field()?;
        let field = match field {
            Some(f) => f,
            None => self
                .default_field
                .map(str::to_string)
                .ok_or(ParseError::MissingField(start))?,
        };

        match self.peek() {
            Some('"') => self.parse_phrase(&field),
            Some('/') => self.parse_regexp(&field),
            Some('[') => self.parse_range(&field),
            Some('{') => Err(ParseError::UnsupportedSyntax(
                self.pos,
                "exclusive range queries ({a TO b}) are not supported".to_string(),
            )),
            None => Err(ParseError::UnexpectedEnd("a term after ':'")),
            _ => self.parse_wordterm(&field),
        }
    }

    /// `'[' bound 'TO' bound ']'` -- an inclusive numeric range, called with
    /// `self.pos` at the opening `[`. `bound` is `*` (open end) or an
    /// optionally-negative decimal `i64`; see the module doc comment for the
    /// exact supported/deferred syntax.
    fn parse_range(&mut self, field: &str) -> Result<Clause, ParseError> {
        let open_pos = self.pos;
        self.advance(); // consume '['
        self.skip_ws();
        let min = self.parse_range_bound(i64::MIN, open_pos)?;
        self.skip_ws();
        self.expect_keyword("TO", open_pos)?;
        self.skip_ws();
        let max = self.parse_range_bound(i64::MAX, open_pos)?;
        self.skip_ws();
        if self.peek() != Some(']') {
            return Err(ParseError::InvalidRangeBound(
                open_pos,
                "expected closing ']'".to_string(),
            ));
        }
        self.advance(); // consume ']'
        Ok(Clause::PointsRange(PointsRangeQuery::new(field, min, max)))
    }

    /// One `[`/`{`-range bound: `*` (mapped to `open_value`) or a plain,
    /// optionally-negative, decimal `i64`.
    fn parse_range_bound(&mut self, open_value: i64, open_pos: usize) -> Result<i64, ParseError> {
        let bound_start = self.pos;
        if self.peek() == Some('*') {
            self.advance();
            return Ok(open_value);
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
        text.parse::<i64>().map_err(|_| {
            ParseError::InvalidRangeBound(open_pos, format!("invalid integer {text:?}"))
        })
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

    /// A bareword: runs of non-whitespace, non-`"/():^` characters (with
    /// `\`-escaping of any byte), optionally followed by `~digits?` (fuzzy).
    /// Decides between `Term`/`Wildcard`/`Prefix`/`Fuzzy` per the module
    /// doc's disambiguation rules.
    fn parse_wordterm(&mut self, field: &str) -> Result<Clause, ParseError> {
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

        // Fuzzy suffix: '~' immediately followed by an optional plain
        // integer (no decimal point -- see module doc's deferred list).
        if self.peek() == Some('~') {
            let tilde_pos = self.pos;
            self.advance();
            let digit_start = self.pos;
            while matches!(self.peek(), Some(c) if c.is_ascii_digit()) {
                self.pos += 1;
            }
            let digits: String = self.chars[digit_start..self.pos].iter().collect();
            // A following '.' means a fractional similarity was requested,
            // which this module doesn't support (see module doc).
            if self.peek() == Some('.') {
                let bad_start = digit_start;
                while matches!(self.peek(), Some(c) if c.is_ascii_digit() || c == '.') {
                    self.pos += 1;
                }
                let text: String = self.chars[bad_start..self.pos].iter().collect();
                return Err(ParseError::InvalidFuzziness(tilde_pos, text));
            }
            let mut fuzzy = FuzzyQuery::new(field, text);
            if !digits.is_empty() {
                let edits: u32 = digits
                    .parse()
                    .map_err(|_| ParseError::InvalidFuzziness(tilde_pos, digits.clone()))?;
                if edits > 2 {
                    return Err(ParseError::InvalidFuzziness(tilde_pos, digits));
                }
                fuzzy = fuzzy.with_max_edits(edits as u8);
            }
            return Ok(Clause::Fuzzy(fuzzy));
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
                return Ok(Clause::Prefix(PrefixQuery::new(field, prefix)));
            }
            return Ok(Clause::Wildcard(WildcardQuery::new(field, wildcard_text)));
        }

        let terms = analyze_term_text(self.analyzer, &text);
        Ok(clause_from_analyzed_terms(field, terms))
    }
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
    fn exclusive_range_query_syntax_is_unsupported() {
        let err = parse_query("body:{0 TO 100}", None).unwrap_err();
        assert!(matches!(err, ParseError::UnsupportedSyntax(_, _)));
    }

    #[test]
    fn bare_exclusive_range_query_syntax_is_unsupported() {
        // The bare-atom `{` arm (no `field:` prefix) is a separate code path
        // from `parse_term`'s -- exercise it directly, not just the
        // field-prefixed sibling above.
        let err = parse_query("{0 TO 100}", Some("body")).unwrap_err();
        assert!(matches!(err, ParseError::UnsupportedSyntax(_, _)));
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
                should: vec![],
                must_not: vec![],
                minimum_should_match: 0,
            }))
        );
    }
}
