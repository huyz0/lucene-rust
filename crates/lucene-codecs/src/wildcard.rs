//! Wildcard/prefix glob matching over already-decoded term bytes — the term
//! side of what real Lucene's `o.a.l.util.automaton` + `WildcardQuery`/
//! `PrefixQuery`/`AutomatonQuery` machinery does when it compiles a pattern
//! into a `CompiledAutomaton`/`ByteRunAutomaton` and drives
//! `IntersectTermsEnum` to walk only the trie blocks that can possibly
//! contain a match.
//!
//! ## Scope of this slice
//!
//! **What's here**: a glob-style pattern (`*` = zero-or-more bytes, `?` =
//! exactly one byte, everything else matches itself literally) plus prefix
//! matching, tested by linear scan over [`crate::blocktree::FieldTerms`]'s
//! already-materialized, already-sorted `entries` `Vec`
//! ([`crate::blocktree::FieldTerms::intersect`]). This covers `PrefixQuery`
//! exactly and `WildcardQuery` for its glob subset (`*`/`?`, no `\`-escaping
//! of literal `*`/`?` — see below).
//!
//! **What's deliberately NOT here** (see `docs/parity.md` for the full
//! writeup):
//!
//! - **No `CompiledAutomaton`/`ByteRunAutomaton` DFA compilation.** Real
//!   Lucene compiles the pattern into a byte-level automaton once and then
//!   drives `IntersectTermsEnum`'s trie walk with it (visiting only the
//!   `.tip`/`.tim` blocks a partial match could reach, potentially skipping
//!   most of a huge dictionary). This module instead tests every candidate
//!   term against the pattern directly — see the module doc for why that's
//!   an honest tradeoff given this port's existing eager-materialization
//!   design, not a placeholder for something better arriving next.
//! - **No true `IntersectTermsEnum` block-skipping.** `FieldTerms::intersect`
//!   scans the field's full sorted `Vec` in `O(n)` (`n` = number of terms in
//!   the field), unlike real Lucene which can be sub-linear in the number of
//!   *matching* terms plus the automaton's own state count. For a prefix
//!   pattern this port narrows the scan with a binary-search range first
//!   (see [`WildcardPattern::literal_prefix`] and `intersect`'s use of it) —
//!   a partial, cheap win that doesn't require automaton machinery, but this
//!   is still fundamentally "filter the sorted Vec," not "walk only the
//!   matching subtree of the trie."
//! - **No regex** (`RegexpQuery`) and **no fuzzy/Levenshtein automaton**
//!   (`FuzzyQuery`). Both need a real automaton/edit-distance representation
//!   (`LevenshteinAutomata`, `RegExp` parsing) that is a materially larger
//!   scope than glob matching over an already-sorted `Vec` — deferred
//!   explicitly rather than half-built here.
//!
//! `\`-escaping of literal `*`/`?` **is** supported (see
//! [`WildcardPattern::new`]'s doc comment) — this mirrors real
//! `WildcardQuery.toAutomaton`'s `WILDCARD_ESCAPE` handling exactly, added
//! for task #34's `WildcardQuery` port.

/// One token of a compiled glob pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    /// A single literal byte that must match exactly.
    Literal(u8),
    /// `?` — exactly one arbitrary Unicode codepoint (1-4 UTF-8 bytes; see
    /// [`WildcardPattern`]'s doc for why this isn't 1 byte).
    AnyOne,
    /// `*` — zero or more arbitrary bytes.
    AnyMany,
}

/// Number of UTF-8 bytes the codepoint starting at `term`'s first byte
/// occupies, or `1` as a fallback when `term` is empty or its first byte
/// isn't a valid UTF-8 lead byte (this port's terms have no guaranteed UTF-8
/// validity — see [`WildcardPattern`]'s doc). Only inspects the leading
/// byte's high bits, the same cheap technique `Character.charCount`'s
/// surrogate-pair check mirrors for UTF-16 -- doesn't validate continuation
/// bytes, so a truncated/invalid multi-byte sequence still advances by the
/// lead byte's nominal width rather than falling back to 1, matching "best
/// effort at the position claimed by the lead byte" rather than full UTF-8
/// validation (this module already doesn't validate term bytes as UTF-8
/// anywhere else).
fn utf8_codepoint_len(term: &[u8]) -> usize {
    match term.first() {
        None => 1,
        Some(&b) if b & 0x80 == 0x00 => 1, // 0xxxxxxx: ASCII
        Some(&b) if b & 0xE0 == 0xC0 => 2, // 110xxxxx
        Some(&b) if b & 0xF0 == 0xE0 => 3, // 1110xxxx
        Some(&b) if b & 0xF8 == 0xF0 => 4, // 11110xxx
        Some(_) => 1,                      // continuation byte or invalid lead: fall back
    }
}

/// A compiled glob-style wildcard pattern over raw term bytes: `*` matches
/// zero or more bytes, `?` matches exactly one Unicode **codepoint** (1-4
/// UTF-8 bytes), every other byte matches itself literally. `?`'s width
/// deliberately isn't 1 byte: real Lucene's `WildcardQuery.toAutomaton`
/// (`WildcardQuery.java`) walks the pattern with `codePointAt`/`charCount`
/// and maps `?` to `Automata.makeAnyChar()` — one arbitrary codepoint, not
/// one encoded byte — so a `?` next to a multi-byte character (e.g. `"a?"`
/// against `"a€"`, `€` = 3 UTF-8 bytes) must consume the whole character to
/// match what real Lucene returns for the equivalent `WildcardQuery`. Terms
/// that aren't valid UTF-8 at the `?`'s position (this port's terms are
/// arbitrary `Vec<u8>`; the wire format makes no UTF-8 guarantee) fall back
/// to consuming exactly 1 byte, since there's no codepoint to align to
/// there.
///
/// No escape syntax is supported (see the module doc): every `*`/`?` in the
/// input is always a wildcard, never a literal.
#[derive(Debug, Clone)]
pub struct WildcardPattern {
    tokens: Vec<Token>,
}

impl WildcardPattern {
    /// Compiles a glob pattern from raw bytes (e.g. `b"foo*"`, `b"a?c"`),
    /// supporting `\` as an escape character exactly the way real Lucene's
    /// `WildcardQuery.toAutomaton` (`WILDCARD_ESCAPE = '\\'`) does: a `\`
    /// followed by another byte forces that following byte to be treated as
    /// a plain literal, even if it's itself `*` or `?` (`\*` matches a
    /// literal `*`, `\?` matches a literal `?`, `\\` matches a literal `\`).
    /// A trailing `\` with nothing after it (no byte left to escape) falls
    /// back to matching a literal `\` itself, mirroring
    /// `WildcardQuery.toAutomaton`'s `case WILDCARD_ESCAPE` fallthrough to
    /// its `default` branch when `i + length >= wildcardText.length()`.
    /// Escaping any byte other than `*`/`?`/`\` is a harmless no-op: that
    /// byte would already have been a plain `Literal` unescaped, same as
    /// real Lucene (escaping a non-special codepoint just re-adds it as
    /// `Automata.makeChar`, identical to the unescaped `default` case).
    pub fn new(pattern: &[u8]) -> Self {
        let mut tokens = Vec::with_capacity(pattern.len());
        let mut i = 0;
        while i < pattern.len() {
            let b = pattern[i];
            if b == b'\\' {
                if let Some(&escaped) = pattern.get(i + 1) {
                    tokens.push(Token::Literal(escaped));
                    i += 2;
                    continue;
                }
                // Trailing, unpaired `\`: literal backslash.
                tokens.push(Token::Literal(b'\\'));
                i += 1;
                continue;
            }
            tokens.push(match b {
                b'*' => Token::AnyMany,
                b'?' => Token::AnyOne,
                other => Token::Literal(other),
            });
            i += 1;
        }
        Self { tokens }
    }

    /// A pure prefix pattern (`PrefixQuery`-equivalent): every term starting
    /// with `prefix` matches, nothing else. Expressed as a `WildcardPattern`
    /// so it shares the same `matches`/`intersect` machinery as glob
    /// patterns (`prefix*` would do the same thing via [`Self::new`]; this
    /// constructor just avoids relying on the caller to append `*` itself).
    pub fn prefix(prefix: &[u8]) -> Self {
        let mut tokens: Vec<Token> = prefix.iter().map(|&b| Token::Literal(b)).collect();
        tokens.push(Token::AnyMany);
        Self { tokens }
    }

    /// Tests whether `term` matches this pattern in full (the entire term
    /// must be consumed — `matches` is not a "contains" or "starts with"
    /// test unless the pattern itself ends in `*`).
    pub fn matches(&self, term: &[u8]) -> bool {
        matches_from(&self.tokens, term)
    }

    /// The pattern's longest literal leading byte run, e.g. `b"foo*ba?"` ->
    /// `b"foo"`, `b"*abc"` -> `b""`, `b"abc"` -> `b"abc"`. Every term this
    /// pattern can match must start with this run, since a `Literal` token
    /// can never match a different byte and `AnyOne`/`AnyMany` only appear
    /// after it — used by [`crate::blocktree::FieldTerms::intersect`] to
    /// narrow its scan to a contiguous sorted range via binary search before
    /// falling back to a per-candidate [`Self::matches`] test, rather than
    /// scanning the field's entire term `Vec` unconditionally.
    pub fn literal_prefix(&self) -> Vec<u8> {
        self.tokens
            .iter()
            .take_while(|t| matches!(t, Token::Literal(_)))
            .map(|t| match t {
                Token::Literal(b) => *b,
                _ => unreachable!("take_while already filtered to Literal"),
            })
            .collect()
    }
}

/// Recursive backtracking glob matcher (`tokens` against `term`). `*` is the
/// only token that can match a variable number of bytes, so the recursion
/// only branches there (try consuming 0, 1, 2, ... bytes of `term` for the
/// `*`, short-circuiting as soon as one branch matches) — every other token
/// advances both `tokens` and `term` by exactly one step (`?` by one
/// codepoint's worth of bytes, everything else by exactly one byte). Worst
/// case this is **multiplicative**, not additive, in the number of `*`
/// tokens: each `*` tries up to `term.len()+1` splits and recurses into the
/// rest of the pattern, which can itself branch at a later `*` -- `O(term.
/// len()^k)` for `k` non-adjacent stars, not `O(k * term.len())`. Real
/// wildcard queries are short and rarely have more than one or two `*`s, so
/// this is not worth replacing with a DP/NFA table for this slice's scope
/// (see the module doc: a real automaton is explicitly deferred), but a
/// pattern with several `*`s against a long term is a genuine, unguarded
/// pathological case, not just a theoretical one.
fn matches_from(tokens: &[Token], term: &[u8]) -> bool {
    match tokens.first() {
        None => term.is_empty(),
        Some(Token::Literal(b)) => match term.first() {
            Some(t) if t == b => matches_from(&tokens[1..], &term[1..]),
            _ => false,
        },
        Some(Token::AnyOne) => {
            if term.is_empty() {
                false
            } else {
                let n = utf8_codepoint_len(term).min(term.len());
                matches_from(&tokens[1..], &term[n..])
            }
        }
        Some(Token::AnyMany) => {
            // Consume the run of consecutive `*`s at once (collapses
            // `**`/`***` to the same behavior as a single `*`, avoiding
            // exponential blowup on adjacent stars) then try every possible
            // split of `term` between "consumed by this `*`" and "left for
            // the rest of the pattern."
            let mut rest_tokens = tokens;
            while matches!(rest_tokens.first(), Some(Token::AnyMany)) {
                rest_tokens = &rest_tokens[1..];
            }
            for split in 0..=term.len() {
                if matches_from(rest_tokens, &term[split..]) {
                    return true;
                }
            }
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str, term: &str) -> bool {
        WildcardPattern::new(pattern.as_bytes()).matches(term.as_bytes())
    }

    #[test]
    fn empty_pattern_matches_only_empty_term() {
        assert!(m("", ""));
        assert!(!m("", "a"));
    }

    #[test]
    fn pure_star_matches_everything_including_empty() {
        assert!(m("*", ""));
        assert!(m("*", "anything"));
        assert!(m("*", "with spaces and \0 bytes"));
    }

    #[test]
    fn literal_pattern_matches_only_itself() {
        assert!(m("term", "term"));
        assert!(!m("term", "terms"));
        assert!(!m("term", "ter"));
        assert!(!m("term", "TERM"));
    }

    #[test]
    fn question_mark_matches_exactly_one_ascii_byte() {
        assert!(m("a?c", "abc"));
        assert!(m("a?c", "axc"));
        assert!(!m("a?c", "ac"));
        assert!(!m("a?c", "abbc"));
        assert!(!m("?", ""));
        assert!(m("?", "x"));
    }

    #[test]
    fn question_mark_matches_one_multi_byte_utf8_codepoint() {
        // Real Lucene's WildcardQuery.toAutomaton maps `?` to one Unicode
        // codepoint via codePointAt/charCount, not one encoded byte -- "a?"
        // must match "a€" (€ = 3 UTF-8 bytes: E2 82 AC) as a single `?`,
        // consuming the whole character, not just its first byte.
        assert!(m("a?", "a€"));
        assert!(!m("a?", "a€x")); // one extra byte left over after the `?`
        assert!(m("a?c", "a€c"));
        assert!(m("?", "𐍈")); // 4-byte codepoint (U+10348)
        assert!(m("?", "é")); // 2-byte codepoint
    }

    #[test]
    fn question_mark_falls_back_to_one_byte_on_invalid_utf8() {
        // A `?` positioned at a non-UTF-8 byte (continuation byte or invalid
        // lead byte) has no codepoint to align to, so it falls back to
        // consuming exactly 1 byte -- confirms the fallback path in
        // `utf8_codepoint_len` rather than panicking or misbehaving.
        let p = WildcardPattern::new(b"a?c");
        assert!(p.matches(&[b'a', 0x80, b'c'])); // 0x80: bare continuation byte
        assert!(p.matches(&[b'a', 0xFF, b'c'])); // 0xFF: never a valid UTF-8 byte
    }

    #[test]
    fn star_suffix_is_prefix_matching() {
        assert!(m("term*", "term"));
        assert!(m("term*", "term0000"));
        assert!(m("term*", "termxyz"));
        assert!(!m("term*", "ter"));
        assert!(!m("term*", "xterm"));
    }

    #[test]
    fn star_prefix_is_suffix_matching() {
        assert!(m("*0000", "term0000"));
        assert!(m("*0000", "0000"));
        assert!(!m("*0000", "0000x"));
    }

    #[test]
    fn star_in_the_middle() {
        assert!(m("term0*99", "term00099"));
        assert!(m("term0*99", "term099"));
        assert!(!m("term0*99", "term098"));
    }

    #[test]
    fn adjacent_stars_collapse_like_one_star() {
        assert!(m("**", "anything"));
        assert!(m("a**b", "ab"));
        assert!(m("a**b", "axxxb"));
        assert!(!m("a**b", "ba"));
    }

    #[test]
    fn matches_nothing_when_no_term_can_satisfy_it() {
        assert!(!m("a?c", "ac"));
        assert!(!m("literal", "totally different"));
        assert!(!m("prefix*suffix", "prefix-only"));
    }

    #[test]
    fn prefix_constructor_matches_prefix_query_semantics() {
        let p = WildcardPattern::prefix(b"term");
        assert!(p.matches(b"term"));
        assert!(p.matches(b"term0000"));
        assert!(!p.matches(b"ter"));
        assert!(!p.matches(b"xterm"));
    }

    #[test]
    fn literal_prefix_extraction() {
        assert_eq!(
            WildcardPattern::new(b"foo*ba?").literal_prefix(),
            b"foo".to_vec()
        );
        assert_eq!(WildcardPattern::new(b"*abc").literal_prefix(), b"".to_vec());
        assert_eq!(
            WildcardPattern::new(b"abc").literal_prefix(),
            b"abc".to_vec()
        );
        assert_eq!(WildcardPattern::new(b"?abc").literal_prefix(), b"".to_vec());
        assert_eq!(WildcardPattern::new(b"").literal_prefix(), b"".to_vec());
    }

    #[test]
    fn escaped_star_and_question_mark_match_only_the_literal_character() {
        assert!(m(r"a\*b", "a*b"));
        assert!(!m(r"a\*b", "axb"));
        assert!(!m(r"a\*b", "ab"));
        assert!(m(r"a\?b", "a?b"));
        assert!(!m(r"a\?b", "axb"));
    }

    #[test]
    fn escaped_backslash_matches_a_literal_backslash() {
        assert!(m(r"a\\b", r"a\b"));
        assert!(!m(r"a\\b", "ab"));
    }

    #[test]
    fn trailing_unescaped_backslash_matches_itself_literally() {
        assert!(m(r"abc\", r"abc\"));
        assert!(!m(r"abc\", "abc"));
    }

    #[test]
    fn escaping_a_non_special_byte_is_a_harmless_no_op() {
        assert!(m(r"a\bc", "abc"));
    }

    #[test]
    fn matches_bytes_that_are_not_valid_utf8() {
        // Terms are arbitrary bytes, not guaranteed UTF-8 -- confirm the
        // matcher operates byte-wise and doesn't assume valid UTF-8.
        let p = WildcardPattern::new(&[b'a', b'*', 0xFF]);
        assert!(p.matches(&[b'a', 0x80, 0xFF]));
        assert!(!p.matches(&[b'a', 0x80, 0xFE]));
    }
}
