//! Wildcard/prefix glob matching over already-decoded term bytes — the term
//! side of what real Lucene's `o.a.l.util.automaton` + `WildcardQuery`/
//! `PrefixQuery`/`AutomatonQuery` machinery does when it compiles a pattern
//! into a `CompiledAutomaton`/`ByteRunAutomaton` and drives
//! `IntersectTermsEnum` to walk only the trie blocks that can possibly
//! contain a match.
//!
//! ## What is here
//!
//! A glob-style pattern (`*` = zero-or-more bytes, `?` = exactly one code
//! point, everything else literal) plus prefix matching, tested by linear scan
//! over [`crate::blocktree::FieldTerms`]'s sorted terms
//! ([`crate::blocktree::FieldTerms::intersect`]). This covers `PrefixQuery`
//! and `WildcardQuery`. A prefix pattern narrows the scan to a binary-searched
//! range first (see [`WildcardPattern::literal_prefix`]). Regular expressions and
//! the edit-distance automaton are in [`crate::regexp`] and [`crate::fuzzy`];
//! the compiled-automaton intersection of the block tree is in
//! [`crate::automaton`].
//!
//! `\`-escaping of literal `*`/`?` **is** supported (see
//! [`WildcardPattern::new`]'s doc comment) — this mirrors real
//! `WildcardQuery.toAutomaton`'s `WILDCARD_ESCAPE` handling exactly.

/// One token of a compiled glob pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    // ARITH: `i` is a byte index into `pattern` held below `pattern.len()` by
    // the loop condition, and a slice length is at most `isize::MAX`, so
    // `i + 1` and `i += 2` cannot reach `usize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
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

    /// A byte DFA accepting a **superset** of this pattern's matches, for
    /// term-dictionary intersection (see `crate::automaton`). Literals and `*`
    /// are exact; `?` is [`crate::automaton::Nfa::lenient_char`], exact for
    /// ASCII and a superset around multi-byte or truncated characters. Every
    /// term it accepts is still confirmed by [`Self::matches`].
    /// [`Self::to_dfa`], compiled once per pattern for the process: a prefix
    /// or wildcard query is expanded in every segment of every search, and the
    /// subset construction behind the automaton was half of a keyword prefix
    /// query's time. Lucene compiles its automaton once per query; this keeps
    /// the last few hundred patterns. Bounded: cleared when full.
    pub(crate) fn to_dfa_cached(&self) -> Option<std::sync::Arc<crate::automaton::ByteDfa>> {
        if let Some(hit) = DFA_CACHE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&self.tokens)
        {
            return hit.clone();
        }
        let dfa = self.to_dfa().map(std::sync::Arc::new);
        let bytes = dfa.as_ref().map_or(0, |d| d.ram_bytes());
        if bytes > MAX_CACHED_BYTES / 4 {
            return dfa;
        }
        let mut cache = DFA_CACHE.lock().unwrap_or_else(|e| e.into_inner());
        let held: usize = cache
            .values()
            .map(|d| d.as_ref().map_or(0, |d| d.ram_bytes()))
            .sum();
        if cache.len() >= MAX_CACHED_PATTERNS || held.saturating_add(bytes) > MAX_CACHED_BYTES {
            cache.clear();
        }
        cache.insert(self.tokens.clone(), dfa.clone());
        dfa
    }

    pub(crate) fn to_dfa(&self) -> Option<crate::automaton::ByteDfa> {
        let mut nfa = crate::automaton::Nfa::new();
        let start = nfa.state()?;
        let mut s = start;
        for token in &self.tokens {
            s = match *token {
                Token::Literal(b) => {
                    let t = nfa.state()?;
                    nfa.range(s, b, b, t);
                    t
                }
                Token::AnyMany => {
                    let hub = nfa.state()?;
                    nfa.epsilon(s, hub);
                    nfa.range(hub, 0x00, 0xFF, hub);
                    hub
                }
                Token::AnyOne => {
                    let t = nfa.state()?;
                    nfa.lenient_char(s, t)?;
                    t
                }
            };
        }
        nfa.determinize(start, s)
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
/// this is not worth replacing with a DP/NFA table here (a real automaton
/// is in [`crate::automaton`]), but a
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

type DfaCache =
    std::collections::HashMap<Vec<Token>, Option<std::sync::Arc<crate::automaton::ByteDfa>>>;

/// The process-wide automata of [`WildcardPattern::to_dfa_cached`], by
/// pattern.
static DFA_CACHE: std::sync::LazyLock<std::sync::Mutex<DfaCache>> =
    std::sync::LazyLock::new(Default::default);

/// Patterns [`DFA_CACHE`] holds before it is cleared.
const MAX_CACHED_PATTERNS: usize = 512;

/// Automaton bytes ([`ByteDfa::ram_bytes`]) [`DFA_CACHE`] holds before it is
/// cleared; an automaton over a quarter of this is used and not kept.
///
/// [`ByteDfa::ram_bytes`]: crate::automaton::ByteDfa::ram_bytes
const MAX_CACHED_BYTES: usize = 16 << 20;

#[cfg(test)]
mod tests;
