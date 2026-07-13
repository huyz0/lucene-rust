//! Regular-expression matching over already-decoded term bytes -- the term
//! side of what real Lucene's `org.apache.lucene.search.RegexpQuery` does
//! when it parses a pattern into an `o.a.l.util.automaton.RegExp` and
//! compiles that into a `CompiledAutomaton`/`ByteRunAutomaton` driving
//! `IntersectTermsEnum`'s trie walk. Structurally this module is the same
//! "match predicate + prefix-narrowed `FieldTerms` scan" shape
//! `fuzzy.rs`/`wildcard.rs` already established (see [`RegexpPattern::
//! literal_prefix`] and `crate::blocktree::FieldTerms::regexp_intersect`).
//!
//! ## Scope decision: hand-built parser, not the `regex` crate (task #43)
//!
//! No crate in this workspace depends on `regex` (checked: no `Cargo.toml`
//! anywhere references it), so this module had a real choice: (a) write a
//! small parser/matcher for Lucene's own restricted `RegExp` syntax, or (b)
//! pull in Rust's `regex` crate and translate/restrict its semantics to
//! Lucene's conventions. **(a) was chosen.** Reasons:
//!
//! - Real Lucene's `RegExp` is deliberately **not** PCRE/Perl regex: it has
//!   no anchors (`^`/`$` are ordinary literal characters in Lucene's
//!   `RegExp`, not zero-width assertions -- irrelevant here anyway, since
//!   `RegexpQuery` always matches a term **in full**, never a substring),
//!   no lookahead/lookbehind, no backreferences, and its own extra
//!   operators (`~` complement, `&` intersection) that standard `regex`
//!   doesn't have at all. Reusing `regex` directly would either silently
//!   accept syntax Lucene rejects (`^`/`$`/lookahead: `regex` would compile
//!   them and search a substring by default) or would require its own
//!   translation/validation layer nearly as large as writing a purpose-built
//!   parser -- with the added risk of a `regex`-specific edge case (e.g.
//!   `regex`'s substring-search-by-default requiring careful `^...$`
//!   wrapping to force whole-match, which is exactly the kind of subtle
//!   divergence the task flagged) leaking through unnoticed.
//! - The already-established `fuzzy.rs`/`wildcard.rs` precedent in this
//!   crate is "hand-build a small matcher scoped to exactly what's needed,
//!   documented honestly," not "reach for an off-the-shelf crate." A
//!   from-scratch backtracking matcher over the concatenation/alternation/
//!   quantifier/class subset below is a natural continuation of that
//!   pattern, not a bigger lift than wiring and constraining `regex` safely
//!   would have been.
//!
//! ## Exact syntax subset supported
//!
//! - Literal bytes match themselves (`\` escapes the following byte to a
//!   plain literal even if it's itself a metacharacter -- same escape
//!   convention `wildcard.rs`'s `WildcardPattern::new` already uses).
//! - `.` -- any single byte (see the "byte, not codepoint" note below).
//! - `*`, `+`, `?` -- postfix quantifiers (zero-or-more / one-or-more /
//!   zero-or-one) on the *immediately preceding* atom (a single literal,
//!   `.`, a `[...]` class, or a parenthesized group).
//! - `[...]` -- a character class: a run of literal bytes and/or `a-z`-style
//!   byte ranges, optionally negated with a leading `^` (`[^abc]`).
//! - `(...)` -- grouping (affects only precedence/quantifier scope and
//!   alternation nesting; does not capture).
//! - `|` -- alternation between the terms on either side, at the current
//!   grouping level.
//!
//! **What's deliberately NOT supported** (real Lucene `RegExp` operators
//! this port does not implement -- rejected with a parse error rather than
//! silently mis-parsed):
//!
//! - `{n}`/`{n,m}` bounded repetition.
//! - `~` (complement) and `&` (intersection) -- Lucene-specific operators
//!   with no direct backtracking-matcher analogue as cheap as the rest of
//!   this subset; a real implementation would need actual automaton
//!   complementation, which is a materially larger undertaking than the
//!   rest of this module (same "defer rather than half-build" call
//!   `wildcard.rs`'s own module doc already made for regex/fuzzy before
//!   this task).
//! - Named classes (`\d`, `\w`, POSIX `[:alpha:]`, etc).
//! - Anchors `^`/`$` as zero-width assertions: since `RegexpQuery` always
//!   matches a term's **entire** length (there is no substring-match mode
//!   at all -- see [`RegexpPattern::matches`]'s doc comment), an anchor
//!   would be redundant even if supported; this module doesn't special-case
//!   `^`/`$` at all, so outside of `[^...]`'s negation position they are
//!   ordinary literal bytes, exactly like real Lucene's `RegExp`.
//!
//! ## Byte-vs-codepoint scope decision
//!
//! Same tradeoff `fuzzy.rs`'s module doc already documents for edit
//! distance: real Lucene's `RegExp`/`Automaton` machinery operates on
//! **Unicode codepoints** (`.` and `[...]` ranges are codepoint-wide), while
//! this module's `.`/class matching is **byte**-wide (terms are `Vec<u8>`
//! with no guaranteed UTF-8 validity). For ASCII terms and patterns -- every
//! fixture this port currently tests against -- one byte and one codepoint
//! coincide, so this is a pragmatic, stated shortcut, not a silent
//! Unicode-correctness claim. A pattern like `.` against a multi-byte UTF-8
//! character would consume only its first byte here, not the whole
//! character the way real Lucene's automaton would.
//!
//! ## Whole-term-match convention
//!
//! Real `RegexpQuery` always matches a term's **entire** length -- there is
//! no partial/substring-match mode, unlike some general-purpose regex
//! engines' default behavior. [`RegexpPattern::matches`] enforces this
//! directly (the backtracking search only succeeds when it consumes the
//! candidate term exactly to its end), so e.g. pattern `ca` does **not**
//! match term `cat` (see this module's `whole_term_match_*` tests) -- the
//! classic "looks right in isolation, subtly wrong vs real regex
//! conventions" bug this task's differential fixture exists to catch.

use std::fmt;

/// A parse error for an unsupported or malformed pattern -- see the module
/// doc's "what's deliberately NOT supported" list for exactly which syntax
/// is rejected here rather than silently mis-parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexpError {
    /// A `(` with no matching `)` before the pattern ended.
    UnmatchedOpenParen,
    /// A `)` with no matching `(` before it.
    UnmatchedCloseParen,
    /// A `[` with no matching `]` before the pattern ended.
    UnmatchedOpenBracket,
    /// A `[...]` class with no bytes in it (`[]` or `[^]`).
    EmptyClass,
    /// A `*`/`+`/`?` with no preceding atom to quantify (e.g. pattern starts
    /// with `*`, or two quantifiers in a row like `a**`).
    DanglingQuantifier,
    /// An operator this port explicitly does not support -- see the module
    /// doc's "what's deliberately NOT supported" list. Carries the
    /// offending byte for a useful error message.
    UnsupportedOperator(u8),
}

impl fmt::Display for RegexpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexpError::UnmatchedOpenParen => write!(f, "unmatched '(' in regexp pattern"),
            RegexpError::UnmatchedCloseParen => write!(f, "unmatched ')' in regexp pattern"),
            RegexpError::UnmatchedOpenBracket => write!(f, "unmatched '[' in regexp pattern"),
            RegexpError::EmptyClass => write!(f, "empty [...] character class in regexp pattern"),
            RegexpError::DanglingQuantifier => {
                write!(f, "quantifier ('*'/'+'/'?') with no preceding atom")
            }
            RegexpError::UnsupportedOperator(b) => write!(
                f,
                "unsupported regexp operator '{}' (byte 0x{b:02x}) -- \
                 this port supports only literals/./*+?/[]/()/| \
                 (no {{n,m}}, no '~' complement, no '&' intersection)",
                *b as char
            ),
        }
    }
}

impl std::error::Error for RegexpError {}

/// One node of a parsed pattern's AST.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    Literal(u8),
    AnyByte,
    Class {
        ranges: Vec<(u8, u8)>,
        negated: bool,
    },
    Concat(Vec<Node>),
    Star(Box<Node>),
    Plus(Box<Node>),
    Ques(Box<Node>),
    Alt(Vec<Node>),
}

/// A compiled Lucene-regexp-subset pattern (see the module doc for exactly
/// which syntax is supported) over raw term bytes. Mirrors `wildcard.rs`'s
/// `WildcardPattern`: a small, cheap-to-build value that
/// [`crate::blocktree::FieldTerms`]'s scanning logic tests every candidate
/// term against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexpPattern {
    root: Node,
}

impl RegexpPattern {
    /// Parses `pattern` (raw bytes -- terms in this port are `Vec<u8>` with
    /// no guaranteed UTF-8 validity, same rationale `WildcardPattern::new`'s
    /// own doc comment gives) into a compiled pattern, or a [`RegexpError`]
    /// if it uses syntax this module doesn't support (see the module doc).
    pub fn new(pattern: &[u8]) -> Result<Self, RegexpError> {
        let mut p = Parser {
            bytes: pattern,
            pos: 0,
        };
        let root = p.parse_alt()?;
        if p.pos < p.bytes.len() {
            // Only reachable via a stray, unmatched ')' at the top level.
            return Err(RegexpError::UnmatchedCloseParen);
        }
        Ok(Self { root })
    }

    /// Tests whether `term` matches this pattern **in full** -- real
    /// `RegexpQuery`'s whole-term-match convention (see the module doc):
    /// e.g. pattern `ca` does not match term `cat`, only term `ca` exactly.
    pub fn matches(&self, term: &[u8]) -> bool {
        node_match(&self.root, term, &|rest| rest.is_empty())
    }

    /// The pattern's longest guaranteed literal leading byte run, e.g.
    /// `cat.*` -> `cat`, `(cat|dog)` -> `` (no single common leading byte
    /// run across an alternation, so this conservatively returns empty
    /// rather than trying to find alternation's common prefix), `c*at` ->
    /// `` (a quantified atom may match zero times, so nothing after it is
    /// guaranteed either -- the leading run stops at the first quantified
    /// atom). Used by [`crate::blocktree::FieldTerms::regexp_intersect`] to
    /// narrow its scan to a contiguous sorted range via binary search first,
    /// the same trick `wildcard.rs`'s `literal_prefix`/`fuzzy.rs`'s
    /// `FuzzyMatch::literal_prefix` already use. Returning an empty `Vec`
    /// (falling back to a full-field scan) is always *correct*, just not
    /// optimized -- this is the documented, acceptable fallback the task
    /// allows when a pattern's leading literal run can't be safely
    /// determined (e.g. starts with `.`, a class, `*`, or an alternation).
    pub fn literal_prefix(&self) -> Vec<u8> {
        // `Parser::parse_alt` always produces either a `Node::Concat`
        // (possibly of zero or one entries, for an empty or single-atom
        // pattern) or a `Node::Alt` (two or more alternatives) as the root
        // -- never a bare `Literal`/`AnyByte`/etc -- so only the `Concat`
        // arm can ever contribute a guaranteed leading literal run; an
        // `Alt` root conservatively returns empty (see this method's doc
        // comment for why alternation has no useful common prefix here).
        let mut prefix = Vec::new();
        let Node::Concat(nodes) = &self.root else {
            return prefix;
        };
        for node in nodes {
            match node {
                Node::Literal(b) => prefix.push(*b),
                _ => break,
            }
        }
        prefix
    }
}

/// Backtracking matcher: does `node` match some prefix of `term`, such that
/// `cont` (given the unconsumed remainder) also succeeds? Continuation-
/// passing lets `Concat`/`Alt`/quantifiers compose correctly across
/// arbitrary nesting (a group's internal choice of how much to consume must
/// be allowed to depend on what comes *after* the group) -- the same shape
/// `wildcard.rs`'s `matches_from` uses for its simpler `*`/`?`-only
/// grammar, generalized here to handle groups and alternation.
fn node_match(node: &Node, term: &[u8], cont: &dyn Fn(&[u8]) -> bool) -> bool {
    match node {
        Node::Literal(b) => term.first() == Some(b) && cont(&term[1..]),
        Node::AnyByte => !term.is_empty() && cont(&term[1..]),
        Node::Class { ranges, negated } => match term.first() {
            Some(&b) => {
                let in_class = ranges.iter().any(|&(lo, hi)| b >= lo && b <= hi);
                if in_class != *negated {
                    cont(&term[1..])
                } else {
                    false
                }
            }
            None => false,
        },
        Node::Concat(nodes) => concat_match(nodes, term, cont),
        Node::Alt(alts) => alts.iter().any(|n| node_match(n, term, cont)),
        Node::Ques(inner) => cont(term) || node_match(inner, term, cont),
        Node::Plus(inner) => node_match(inner, term, &|rest| star_match(inner, rest, cont)),
        Node::Star(inner) => star_match(inner, term, cont),
    }
}

/// `inner*` against `term`: try the shortest match first (zero repetitions,
/// i.e. `cont(term)` directly), then one-or-more, guarding against an
/// infinite loop on a zero-width repetition (an inner node that can match
/// while consuming no bytes at all) by refusing to recurse when a
/// repetition made no progress.
fn star_match(inner: &Node, term: &[u8], cont: &dyn Fn(&[u8]) -> bool) -> bool {
    if cont(term) {
        return true;
    }
    node_match(inner, term, &|rest| {
        if rest.len() == term.len() {
            // No progress this iteration -- would recurse forever on a
            // zero-width inner match; every atom in this module's grammar
            // always consumes exactly one byte per repetition (`Literal`/
            // `AnyByte`/`Class`), so this only guards a pathological/
            // future case rather than firing in practice.
            false
        } else {
            star_match(inner, rest, cont)
        }
    })
}

fn concat_match(nodes: &[Node], term: &[u8], cont: &dyn Fn(&[u8]) -> bool) -> bool {
    match nodes.split_first() {
        None => cont(term),
        Some((first, rest)) => node_match(first, term, &|r| concat_match(rest, r, cont)),
    }
}

/// Recursive-descent parser for the syntax subset in this module's doc
/// comment. Grammar (informal):
///
/// ```text
/// alt    := concat ('|' concat)*
/// concat := factor*
/// factor := atom ('*' | '+' | '?')?
/// atom   := literal | '.' | '[' class ']' | '(' alt ')'
/// class  := '^'? item+
/// item   := byte | byte '-' byte
/// ```
struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<u8> {
        let b = self.peek();
        if b.is_some() {
            self.pos += 1;
        }
        b
    }

    fn parse_alt(&mut self) -> Result<Node, RegexpError> {
        let mut alts = vec![self.parse_concat()?];
        while self.peek() == Some(b'|') {
            self.advance();
            alts.push(self.parse_concat()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().unwrap()
        } else {
            Node::Alt(alts)
        })
    }

    fn parse_concat(&mut self) -> Result<Node, RegexpError> {
        let mut nodes = Vec::new();
        while let Some(b) = self.peek() {
            if b == b'|' || b == b')' {
                break;
            }
            nodes.push(self.parse_factor()?);
        }
        Ok(Node::Concat(nodes))
    }

    fn parse_factor(&mut self) -> Result<Node, RegexpError> {
        let atom = self.parse_atom()?;
        match self.peek() {
            Some(b'*') => {
                self.advance();
                Ok(Node::Star(Box::new(atom)))
            }
            Some(b'+') => {
                self.advance();
                Ok(Node::Plus(Box::new(atom)))
            }
            Some(b'?') => {
                self.advance();
                Ok(Node::Ques(Box::new(atom)))
            }
            _ => Ok(atom),
        }
    }

    fn parse_atom(&mut self) -> Result<Node, RegexpError> {
        let b = self.advance().ok_or(RegexpError::DanglingQuantifier)?;
        match b {
            b'*' | b'+' | b'?' => Err(RegexpError::DanglingQuantifier),
            b'{' | b'~' | b'&' => Err(RegexpError::UnsupportedOperator(b)),
            b'\\' => {
                let escaped = self.advance().unwrap_or(b'\\');
                Ok(Node::Literal(escaped))
            }
            b'.' => Ok(Node::AnyByte),
            b'(' => {
                let inner = self.parse_alt()?;
                if self.advance() != Some(b')') {
                    return Err(RegexpError::UnmatchedOpenParen);
                }
                Ok(inner)
            }
            b'[' => self.parse_class(),
            // A bare ')' can never reach `parse_atom`: `parse_concat`'s loop
            // (this parser's only caller of `parse_factor`/`parse_atom`)
            // stops before consuming a ')', so `other` below only ever sees
            // an ordinary literal byte, never one of the operators already
            // matched above it.
            other => Ok(Node::Literal(other)),
        }
    }

    fn parse_class(&mut self) -> Result<Node, RegexpError> {
        let negated = if self.peek() == Some(b'^') {
            self.advance();
            true
        } else {
            false
        };
        let mut ranges = Vec::new();
        loop {
            match self.peek() {
                None => return Err(RegexpError::UnmatchedOpenBracket),
                Some(b']') => {
                    self.advance();
                    break;
                }
                Some(b'\\') => {
                    self.advance();
                    let lo = self.advance().ok_or(RegexpError::UnmatchedOpenBracket)?;
                    ranges.push(self.finish_class_item(lo)?);
                }
                Some(lo) => {
                    self.advance();
                    ranges.push(self.finish_class_item(lo)?);
                }
            }
        }
        if ranges.is_empty() {
            return Err(RegexpError::EmptyClass);
        }
        Ok(Node::Class { ranges, negated })
    }

    /// Given a class item's first byte `lo`, checks for a following `-hi`
    /// range suffix (`a-z`); otherwise the item is the single byte `lo`.
    fn finish_class_item(&mut self, lo: u8) -> Result<(u8, u8), RegexpError> {
        if self.peek() == Some(b'-') {
            // Only treat `-` as a range dash when there's a byte after it
            // that isn't the closing `]` (a trailing `-` right before `]`,
            // e.g. `[a-]`, is a literal `-` -- matches the common
            // POSIX-class convention of `-` at the end being literal).
            if let Some(hi) = self.bytes.get(self.pos + 1).copied() {
                if hi != b']' {
                    self.advance(); // consume '-'
                    self.advance(); // consume hi
                    return Ok((lo, hi));
                }
            }
        }
        Ok((lo, lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str, term: &str) -> bool {
        RegexpPattern::new(pattern.as_bytes())
            .unwrap()
            .matches(term.as_bytes())
    }

    #[test]
    fn literal_pattern_matches_only_itself() {
        assert!(m("cat", "cat"));
        assert!(!m("cat", "cats"));
        assert!(!m("cat", "ca"));
        assert!(!m("cat", "CAT"));
    }

    #[test]
    fn whole_term_match_is_enforced_not_substring() {
        // The exact convention the task called out: `ca` must NOT match
        // `cat` as a substring, since RegexpQuery always matches the whole
        // term.
        assert!(!m("ca", "cat"));
        assert!(m("ca", "ca"));
        assert!(!m("at", "cat"));
    }

    #[test]
    fn dot_matches_any_single_byte() {
        assert!(m("c.t", "cat"));
        assert!(m("c.t", "cot"));
        assert!(!m("c.t", "ct"));
        assert!(!m("c.t", "caat"));
    }

    #[test]
    fn star_is_zero_or_more_of_preceding_atom() {
        assert!(m("ca*t", "ct"));
        assert!(m("ca*t", "cat"));
        assert!(m("ca*t", "caaaat"));
        assert!(!m("ca*t", "cbt"));
    }

    #[test]
    fn plus_is_one_or_more_of_preceding_atom() {
        assert!(!m("ca+t", "ct"));
        assert!(m("ca+t", "cat"));
        assert!(m("ca+t", "caaaat"));
    }

    #[test]
    fn question_mark_is_zero_or_one_of_preceding_atom() {
        assert!(m("ca?t", "ct"));
        assert!(m("ca?t", "cat"));
        assert!(!m("ca?t", "caat"));
    }

    #[test]
    fn character_class_matches_any_listed_byte() {
        assert!(m("[cb]at", "cat"));
        assert!(m("[cb]at", "bat"));
        assert!(!m("[cb]at", "hat"));
    }

    #[test]
    fn character_class_range_matches_any_byte_in_range() {
        assert!(m("[a-c]at", "aat"));
        assert!(m("[a-c]at", "bat"));
        assert!(m("[a-c]at", "cat"));
        assert!(!m("[a-c]at", "dat"));
    }

    #[test]
    fn negated_character_class_matches_any_byte_not_listed() {
        assert!(m("[^ab]at", "cat"));
        assert!(!m("[^ab]at", "aat"));
        assert!(!m("[^ab]at", "bat"));
    }

    #[test]
    fn trailing_dash_in_class_is_a_literal_dash() {
        assert!(m("[a-]t", "at"));
        assert!(m("[a-]t", "-t"));
        assert!(!m("[a-]t", "bt"));
    }

    #[test]
    fn escaped_byte_inside_class_is_a_literal_class_member() {
        // `\]` inside a class escapes the closing bracket to a plain
        // literal class member, rather than ending the class early.
        assert!(RegexpPattern::new(br"[\]]").unwrap().matches(b"]"));
        assert!(!RegexpPattern::new(br"[\]]").unwrap().matches(b"x"));
    }

    #[test]
    fn trailing_unmatched_backslash_inside_class_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"[a\\").unwrap_err(),
            RegexpError::UnmatchedOpenBracket
        );
    }

    #[test]
    fn alternation_matches_either_side() {
        assert!(m("cat|dog", "cat"));
        assert!(m("cat|dog", "dog"));
        assert!(!m("cat|dog", "bird"));
    }

    #[test]
    fn grouping_scopes_quantifiers_and_alternation() {
        assert!(m("(cat)+", "catcat"));
        assert!(!m("(cat)+", "ca"));
        assert!(m("(cat|dog)s", "cats"));
        assert!(m("(cat|dog)s", "dogs"));
        assert!(!m("(cat|dog)s", "birds"));
    }

    #[test]
    fn nested_groups_and_alternation_compose() {
        assert!(m("(a(b|c)d)+", "abdacd"));
        assert!(!m("(a(b|c)d)+", "abdaed"));
    }

    #[test]
    fn no_match_case() {
        assert!(!m("cat", "dog"));
        assert!(!m("c[ao]t", "cbt"));
    }

    #[test]
    fn escaped_metacharacter_is_a_literal() {
        assert!(m(r"a\*b", "a*b"));
        assert!(!m(r"a\*b", "aab"));
        assert!(m(r"a\.b", "a.b"));
        assert!(!m(r"a\.b", "axb"));
    }

    #[test]
    fn escaped_backslash_and_bracket_are_literals() {
        assert!(m(r"a\\b", r"a\b"));
        assert!(m(r"a\[b", "a[b"));
    }

    #[test]
    fn trailing_unescaped_backslash_matches_itself() {
        assert!(m(r"ab\", r"ab\"));
    }

    #[test]
    fn empty_pattern_matches_only_empty_term() {
        assert!(m("", ""));
        assert!(!m("", "a"));
    }

    #[test]
    fn unmatched_open_paren_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"(cat").unwrap_err(),
            RegexpError::UnmatchedOpenParen
        );
    }

    #[test]
    fn unmatched_close_paren_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"cat)").unwrap_err(),
            RegexpError::UnmatchedCloseParen
        );
    }

    #[test]
    fn unmatched_open_bracket_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"[cat").unwrap_err(),
            RegexpError::UnmatchedOpenBracket
        );
    }

    #[test]
    fn empty_class_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"[]").unwrap_err(),
            RegexpError::EmptyClass
        );
        assert_eq!(
            RegexpPattern::new(b"[^]").unwrap_err(),
            RegexpError::EmptyClass
        );
    }

    #[test]
    fn dangling_quantifier_is_a_parse_error() {
        assert_eq!(
            RegexpPattern::new(b"*cat").unwrap_err(),
            RegexpError::DanglingQuantifier
        );
        assert_eq!(
            RegexpPattern::new(b"a**").unwrap_err(),
            RegexpError::DanglingQuantifier
        );
    }

    #[test]
    fn unsupported_operators_are_rejected_not_silently_mismatched() {
        assert_eq!(
            RegexpPattern::new(b"a{2,3}").unwrap_err(),
            RegexpError::UnsupportedOperator(b'{')
        );
        assert_eq!(
            RegexpPattern::new(b"a~b").unwrap_err(),
            RegexpError::UnsupportedOperator(b'~')
        );
        assert_eq!(
            RegexpPattern::new(b"a&b").unwrap_err(),
            RegexpError::UnsupportedOperator(b'&')
        );
    }

    #[test]
    fn error_display_mentions_supported_subset() {
        let msg = RegexpError::UnsupportedOperator(b'{').to_string();
        assert!(msg.contains("no {n,m}"));
    }

    #[test]
    fn every_error_variant_has_a_non_empty_display_message() {
        // Not a semantic check on the exact wording (that's
        // `error_display_mentions_supported_subset` above) -- just
        // confirms every `Display` arm actually runs and produces
        // something, since a `Debug`-derived `assert_eq!` in the parse-
        // error tests above never exercises `Display` itself.
        for err in [
            RegexpError::UnmatchedOpenParen,
            RegexpError::UnmatchedCloseParen,
            RegexpError::UnmatchedOpenBracket,
            RegexpError::EmptyClass,
            RegexpError::DanglingQuantifier,
            RegexpError::UnsupportedOperator(b'~'),
        ] {
            assert!(!err.to_string().is_empty());
        }
    }

    #[test]
    fn literal_prefix_of_pure_literal_pattern_is_the_whole_pattern() {
        assert_eq!(
            RegexpPattern::new(b"cat").unwrap().literal_prefix(),
            b"cat".to_vec()
        );
    }

    #[test]
    fn literal_prefix_stops_at_first_non_literal_atom() {
        assert_eq!(
            RegexpPattern::new(b"ca.*").unwrap().literal_prefix(),
            b"ca".to_vec()
        );
        assert_eq!(
            RegexpPattern::new(b"ca*t").unwrap().literal_prefix(),
            b"c".to_vec()
        );
    }

    #[test]
    fn literal_prefix_of_alternation_is_empty() {
        assert_eq!(
            RegexpPattern::new(b"cat|dog").unwrap().literal_prefix(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn literal_prefix_of_leading_wildcard_atom_is_empty() {
        assert_eq!(
            RegexpPattern::new(b".*cat").unwrap().literal_prefix(),
            Vec::<u8>::new()
        );
        assert_eq!(
            RegexpPattern::new(b"[ab]cat").unwrap().literal_prefix(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn literal_prefix_of_empty_pattern_is_empty() {
        assert_eq!(
            RegexpPattern::new(b"").unwrap().literal_prefix(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn single_bare_literal_pattern_has_a_one_byte_prefix() {
        // `parse_concat` always wraps in `Node::Concat`, even a single-atom
        // one, so this exercises the `Node::Concat` branch with one entry --
        // included to also document the bare non-`Concat` fallback branch
        // stays correct if a future refactor ever bypasses `Concat`.
        assert_eq!(
            RegexpPattern::new(b"c").unwrap().literal_prefix(),
            b"c".to_vec()
        );
    }
}
