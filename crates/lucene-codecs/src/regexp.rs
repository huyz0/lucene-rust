//! Regular-expression matching over already-decoded term bytes -- the term
//! side of what real Lucene's `org.apache.lucene.search.RegexpQuery` does
//! when it parses a pattern into an `o.a.l.util.automaton.RegExp` and
//! compiles that into a `CompiledAutomaton`/`ByteRunAutomaton` driving
//! `IntersectTermsEnum`'s trie walk. Structurally this module is the same
//! "match predicate + prefix-narrowed `FieldTerms` scan" shape
//! `fuzzy.rs`/`wildcard.rs` already established (see [`RegexpPattern::
//! literal_prefix`] and `crate::blocktree::FieldTerms::regexp_intersect`).
//!
//! ## Grammar: Lucene's `RegExp`, not PCRE
//!
//! The parser below implements `o.a.l.util.automaton.RegExp`'s **whole**
//! grammar as `RegexpQuery(Term)` enables it -- `RegExp.ALL` (`0xff`) syntax
//! flags, zero match flags -- production by production. `RegExp.ALL` turns on
//! `INTERSECTION | EMPTY | ANYSTRING | AUTOMATON | INTERVAL`; the
//! `DEPRECATED_COMPLEMENT` flag (`0x10000`) is **not** part of `ALL`, so `~`
//! is an ordinary literal character, exactly as it is in real Lucene when a
//! caller uses the default constructor.
//!
//! | Syntax | Meaning | Lucene flag |
//! |---|---|---|
//! | `a` | the literal codepoint `a` | always |
//! | `\a` | escaped literal (`\` before a non-alphabetic codepoint) | always |
//! | `.` | any single codepoint (`Automata.makeAnyChar`) | always |
//! | `"..."` | literal string, no escapes honoured inside | always |
//! | `()` | the empty string | always |
//! | `(e)` | grouping | always |
//! | `e1\|e2` | union | always |
//! | `e1&e2` | intersection | `INTERSECTION` |
//! | `e?` `e*` `e+` | optional / repeat / repeat-at-least-once | always |
//! | `e{n}` `e{n,}` `e{n,m}` | bounded repetition | always |
//! | `[abc]` `[a-z]` `[^a-z]` | character class over codepoints | always |
//! | `\d \D \s \S \w \W \\` | predefined classes, inside or outside `[...]` | always |
//! | `#` | the empty language (matches nothing) | `EMPTY` |
//! | `@` | any string | `ANYSTRING` |
//! | `<n-m>` | numeric interval | `INTERVAL` |
//! | `<identifier>` | named automaton | `AUTOMATON` |
//!
//! Named automata (`<identifier>`) parse, then fail at compile time with
//! [`RegexpError::NamedAutomatonNotFound`] -- real Lucene behaves identically
//! for `RegexpQuery`, whose `DEFAULT_PROVIDER` returns `null` for every name
//! and whose `RegExp.toAutomaton` then throws
//! `IllegalArgumentException("'" + s + "' not found")`.
//!
//! Deliberately **not** supported, and the only two gaps left:
//!
//! - `RegExp.CASE_INSENSITIVE` / `CASE_INSENSITIVE_RANGE` **match** flags.
//!   `RegexpQuery(Term)` passes match flags `0`, so this only affects a
//!   caller who would have constructed `RegexpQuery(term, syntaxFlags,
//!   matchFlags, ...)` explicitly -- and this port has no such API surface.
//! - `RegExp.DEPRECATED_COMPLEMENT` (`~`). Not in `RegExp.ALL`, removed
//!   outright in Lucene 11, and `[^...]` covers what it was used for.
//!
//! ## Codepoints, not bytes
//!
//! Real Lucene builds a **UTF-32 codepoint** automaton and then converts it
//! to a UTF-8 byte automaton with `UTF32ToUTF8` before running it over term
//! bytes. This module skips the conversion and decodes the candidate term's
//! UTF-8 on the fly instead, which gives the same answer: `.`, `[...]` and a
//! literal all consume exactly one **codepoint**, so `.` against `"€"` (3
//! UTF-8 bytes) matches, and `[^a]` against `"€"` matches.
//!
//! A term byte sequence that is **not** valid UTF-8 matches nothing, which is
//! also what real Lucene does: `UTF32ToUTF8` can only ever emit well-formed
//! UTF-8 byte sequences, so no compiled `RegExp` automaton accepts an
//! ill-formed one. The pattern itself is decoded leniently
//! (`String::from_utf8_lossy`), since it reaches this module from a `String`
//! in every caller.
//!
//! ## Whole-term-match convention
//!
//! Real `RegexpQuery` always matches a term's **entire** length -- there is
//! no partial/substring-match mode, unlike some general-purpose regex
//! engines' default behavior. [`RegexpPattern::matches`] enforces this
//! directly (the backtracking search only succeeds when it consumes the
//! candidate term exactly to its end), so e.g. pattern `ca` does **not**
//! match term `cat` (see this module's `whole_term_match_*` tests).
//!
//! ## Why a backtracker and not an `Automaton`
//!
//! Real Lucene determinizes the parsed `RegExp` into a byte-level DFA and
//! runs it with `ByteRunAutomaton`, which is what makes `IntersectTermsEnum`
//! able to *skip* term-dictionary blocks. This module instead evaluates the
//! parsed tree directly with a bounded backtracking search. That is a
//! deliberate, recorded scope decision -- see `docs/sweep/m2/
//! b8-automata-analysis.md` -- and the cost is scan volume, not correctness:
//! `crate::blocktree::FieldTerms::regexp_intersect` tests every term in the
//! [`RegexpPattern::literal_prefix`] range rather than only the terms a DFA
//! could reach.

use std::cell::Cell;
use std::fmt;

/// Hard ceiling on total backtracking steps `RegexpPattern::matches` will
/// spend on a single term, regardless of pattern shape -- see that method's
/// doc comment for why bounded repetition (`{n,m}`) makes this necessary.
/// Chosen generously above any realistic legitimate match (worst-case
/// legitimate patterns in this module's test suite spend well under 10_000
/// steps) while still bounding a pathological nested-repeat pattern to a
/// bounded, sub-second amount of work. Real Lucene's equivalent guard is
/// `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT`, which caps the *construction*
/// of the DFA rather than the matching, and throws
/// `TooComplexToDeterminizeException` instead of reporting "no match".
const MATCH_STEP_BUDGET: u64 = 1_000_000;

/// The largest legal Unicode codepoint -- `Character.MAX_CODE_POINT`, the
/// upper bound of `Automata.makeAnyChar()`'s single transition.
const MAX_CODE_POINT: u32 = 0x10_FFFF;

/// A parse (or compile) error for a malformed pattern. Each variant mirrors
/// one `IllegalArgumentException` real Lucene's `RegExp` parser throws, with
/// the same triggering condition; `pos` is the codepoint index into the
/// pattern at which Lucene reports the same failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RegexpError {
    /// Java: `"unexpected end-of-string"` -- `RegExp.next()` ran off the end,
    /// e.g. a pattern ending in a lone `\`.
    UnexpectedEndOfString,
    /// Java: `"expected '<c>' at position <pos>"` -- a `)`, `]`, `"`, `>` or
    /// `}` the grammar required was missing.
    ExpectedChar { expected: char, pos: usize },
    /// Java: `"integer expected at position <pos>"` -- a `{` repetition with
    /// no digits after it.
    IntegerExpected { pos: usize },
    /// Java: `"invalid repetition range(out of order): n..m"`.
    InvalidRepetitionRange { min: u32, max: u32 },
    /// Java: `"invalid character class \\<c>"` -- a `\` before an alphabetic
    /// codepoint that is not one of `d D s S w W`.
    InvalidCharacterClass(char),
    /// Java: `"interval syntax error at position <pos>"` -- a `<n-m>` whose
    /// bounds are not two non-negative decimal integers separated by exactly
    /// one `-`.
    IntervalSyntax { pos: usize },
    /// Java: `"illegal identifier at position <pos>"` -- reserved for
    /// `<n-m>` when the `INTERVAL` flag is off. Unreachable through
    /// [`RegexpPattern::new`] (which is always `RegExp.ALL`), kept so the
    /// error surface matches Lucene's one-for-one.
    IllegalIdentifier { pos: usize },
    /// Java: `"'<name>' not found"` -- `<identifier>` with no
    /// `AutomatonProvider` that knows the name, which is every case for
    /// `RegexpQuery`, whose `DEFAULT_PROVIDER` always returns `null`.
    NamedAutomatonNotFound(String),
    /// Java: `"end-of-string expected at position <pos>"` -- trailing input
    /// the grammar could not consume, i.e. an unmatched `)`.
    EndOfStringExpected { pos: usize },
}

impl fmt::Display for RegexpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegexpError::UnexpectedEndOfString => write!(f, "unexpected end-of-string"),
            RegexpError::ExpectedChar { expected, pos } => {
                write!(f, "expected '{expected}' at position {pos}")
            }
            RegexpError::IntegerExpected { pos } => {
                write!(f, "integer expected at position {pos}")
            }
            RegexpError::InvalidRepetitionRange { min, max } => {
                write!(f, "invalid repetition range(out of order): {min}..{max}")
            }
            RegexpError::InvalidCharacterClass(c) => {
                write!(f, "invalid character class \\{c}")
            }
            RegexpError::IntervalSyntax { pos } => {
                write!(f, "interval syntax error at position {pos}")
            }
            RegexpError::IllegalIdentifier { pos } => {
                write!(f, "illegal identifier at position {pos}")
            }
            RegexpError::NamedAutomatonNotFound(name) => write!(f, "'{name}' not found"),
            RegexpError::EndOfStringExpected { pos } => {
                write!(f, "end-of-string expected at position {pos}")
            }
        }
    }
}

impl std::error::Error for RegexpError {}

/// One node of a parsed pattern's tree. Kinds correspond one-to-one to
/// `RegExp.Kind`, minus the deprecated complement and with `REGEXP_OPTIONAL`/
/// `REGEXP_REPEAT`/`REGEXP_REPEAT_MIN`/`REGEXP_REPEAT_MINMAX` folded into one
/// [`Node::Repeat`] (they differ only in their `min`/`max` bounds) and
/// `REGEXP_CHAR_RANGE` folded into [`Node::Class`] (a one-range class).
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    /// `#` -- `REGEXP_EMPTY`, the empty language: matches nothing at all.
    Empty,
    /// `.` -- `REGEXP_ANYCHAR`.
    AnyChar,
    /// A single literal codepoint -- `REGEXP_CHAR`.
    Char(char),
    /// `"..."`, `()` and the string form of a group -- `REGEXP_STRING`.
    Str(String),
    /// `[...]` -- `REGEXP_CHAR_CLASS`/`REGEXP_CHAR_RANGE`, inclusive
    /// codepoint ranges. `negated` is Lucene's `[^...]`, built there as
    /// `anyChar & complement(class)`, i.e. still exactly one codepoint wide.
    Class {
        ranges: Vec<(u32, u32)>,
        negated: bool,
    },
    /// `REGEXP_CONCATENATION`.
    Concat(Vec<Node>),
    /// `|` -- `REGEXP_UNION`.
    Alt(Vec<Node>),
    /// `&` -- `REGEXP_INTERSECTION`.
    Intersect(Box<Node>, Box<Node>),
    /// `?`/`*`/`+`/`{n}`/`{n,}`/`{n,m}` and `@` (as `AnyChar{0,}`).
    /// `max == None` means unbounded.
    Repeat {
        inner: Box<Node>,
        min: u32,
        max: Option<u32>,
    },
    /// `<n-m>` -- `REGEXP_INTERVAL`. `digits > 0` forces a fixed, zero-padded
    /// width (Lucene sets it when both bounds were written with the same
    /// number of digits); `digits == 0` accepts any width, leading zeros
    /// included. See `Automata.makeDecimalInterval`.
    Interval { min: u32, max: u32, digits: usize },
}

/// A compiled Lucene `RegExp` pattern (see the module doc for the grammar)
/// over raw term bytes. Mirrors `wildcard.rs`'s `WildcardPattern`: a small,
/// cheap-to-build value that [`crate::blocktree::FieldTerms`]'s scanning
/// logic tests every candidate term against.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegexpPattern {
    root: Node,
}

impl RegexpPattern {
    /// Parses `pattern` into a compiled pattern, or a [`RegexpError`] for
    /// exactly the inputs real Lucene's
    /// `new RegExp(text, RegExp.ALL).toAutomaton(RegexpQuery.DEFAULT_PROVIDER)`
    /// rejects.
    ///
    /// `pattern` is taken as bytes because every term-side API in this crate
    /// is byte-oriented, but it is a *string* in Lucene and in every caller
    /// here, so it is decoded as UTF-8 with the usual lossy replacement of
    /// ill-formed sequences before parsing.
    pub fn new(pattern: &[u8]) -> Result<Self, RegexpError> {
        Self::parse(&String::from_utf8_lossy(pattern))
    }

    /// [`Self::new`] for a pattern already known to be a string. This is the
    /// shape `RegExp`'s own constructor has; [`std::str::FromStr`] is
    /// implemented in terms of it, so `"cat.*".parse::<RegexpPattern>()`
    /// works too.
    pub fn parse(pattern: &str) -> Result<Self, RegexpError> {
        // `RegExp`'s constructor short-circuits the empty pattern to
        // `makeString("")` rather than running the parser over it.
        if pattern.is_empty() {
            return Ok(Self {
                root: Node::Str(String::new()),
            });
        }
        let chars: Vec<char> = pattern.chars().collect();
        let mut p = Parser {
            chars: &chars,
            pos: 0,
        };
        let root = p.parse_union()?;
        if p.pos < p.chars.len() {
            // Only reachable via a stray, unmatched ')' at the top level.
            return Err(RegexpError::EndOfStringExpected { pos: p.pos });
        }
        Ok(Self { root })
    }

    /// Tests whether `term` matches this pattern **in full** -- real
    /// `RegexpQuery`'s whole-term-match convention (see the module doc):
    /// e.g. pattern `ca` does not match term `cat`, only term `ca` exactly.
    pub fn matches(&self, term: &[u8]) -> bool {
        // A step budget, not just a nesting-depth cap: bounded repetition
        // (`{n,m}`) lets small, innocent-looking numbers combine
        // multiplicatively when nested (e.g. `(a{1,15}){1,15}` against a
        // long run of `a`s with no trailing match), and `RegexpQuery`
        // patterns can come from an untrusted query string, so this
        // backtracking matcher needs a hard ceiling on total work rather
        // than trusting every pattern to terminate promptly. Exceeding the
        // budget is treated as "no match" (never a panic or a hang) -- the
        // same fail-safe direction real Lucene's automaton construction
        // takes when a pattern would blow up (it rejects it up front
        // instead of exploring it at query time).
        let budget = Cell::new(MATCH_STEP_BUDGET);
        node_match(&self.root, term, &budget, &|rest| rest.is_empty())
    }

    /// The length of the shortest prefix of `term` that no matching term can
    /// begin with, or `None` when every prefix of `term` is still viable.
    ///
    /// This is the capability a `CompiledAutomaton`/`ByteRunAutomaton` gives
    /// `IntersectTermsEnum` and that this port previously had no way to
    /// express: *the automaton is in a dead state, so stop*. Over a sorted
    /// term array it turns into a skip -- when `dead_prefix_len(term)` is
    /// `Some(k)`, every term sharing `term[..k]` is a guaranteed non-match, so
    /// a scan can binary-search straight past the whole run instead of testing
    /// each one. `crate::blocktree::FieldTerms::regexp_intersect` does exactly
    /// that.
    ///
    /// Viability is monotone -- if no term can start with `term[..k]`, none
    /// can start with any longer prefix either -- so the shortest dead prefix
    /// is found by binary search over the length, in `O(log n)` matcher runs
    /// rather than `n`.
    ///
    /// The underlying test is a sound over-approximation of "some extension of
    /// this prefix matches", so a `None` here never means a missed skip that
    /// would have been wrong; it only ever means less pruning.
    pub fn dead_prefix_len(&self, term: &[u8]) -> Option<usize> {
        if self.could_match_prefix(term) {
            return None;
        }
        // A pattern that accepts nothing at all (`#`) kills even the empty
        // prefix, which the binary search below assumes is alive.
        if !self.could_match_prefix(&[]) {
            return Some(0);
        }
        // Invariant: `lo` is viable (the empty prefix always is, since the
        // pattern would then have to reject every term), `hi` is dead.
        let (mut lo, mut hi) = (0usize, term.len());
        // ARITH: `lo < hi <= term.len() <= isize::MAX` throughout (`lo` only
        // moves up to `mid < hi` and `hi` only down to `mid > lo`), so
        // `hi - lo` cannot underflow and `lo + (hi - lo) / 2` cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            // Only split on a UTF-8 boundary: a dead prefix that cuts a
            // codepoint in half would skip terms that share half a character.
            let mid = (mid..=term.len())
                .find(|&i| i == term.len() || (term[i] & 0xC0) != 0x80)
                .unwrap_or(term.len());
            if mid >= hi {
                break;
            }
            if self.could_match_prefix(&term[..mid]) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(hi)
    }

    /// Whether **some** term beginning with `prefix` could match. See
    /// [`Self::dead_prefix_len`]; `false` here is the dead-state signal.
    pub fn could_match_prefix(&self, prefix: &[u8]) -> bool {
        let budget = Cell::new(MATCH_STEP_BUDGET);
        node_prefix_match(&self.root, prefix, &budget, &|rest| rest.is_empty())
    }

    /// The longest byte run every matching term is guaranteed to start with,
    /// e.g. `cat.*` -> `cat`, `(cat|dog)` -> `` (no single common leading
    /// byte run across an alternation, so this conservatively returns empty
    /// rather than computing alternation's common prefix), `c*at` -> `` (a
    /// `{0,n}` atom may match zero times, so nothing after it is guaranteed
    /// either). Used by
    /// [`crate::blocktree::FieldTerms::regexp_intersect`] to narrow its scan
    /// to a contiguous sorted range via binary search first, the same trick
    /// `wildcard.rs`'s `literal_prefix`/`fuzzy.rs`'s
    /// `FuzzyMatch::literal_prefix` already use. This is real Lucene's
    /// `CompiledAutomaton.commonPrefix`, computed from the parse tree here
    /// rather than from a determinized automaton, so it is allowed to be
    /// *shorter* than Lucene's (never longer): returning an empty `Vec` and
    /// falling back to a full-field scan is always correct, just not
    /// optimized.
    pub fn literal_prefix(&self) -> Vec<u8> {
        let mut prefix = Vec::new();
        node_prefix(&self.root, &mut prefix);
        prefix
    }
}

impl std::str::FromStr for RegexpPattern {
    type Err = RegexpError;

    fn from_str(pattern: &str) -> Result<Self, Self::Err> {
        Self::parse(pattern)
    }
}

/// Appends to `out` the literal byte run `node` guarantees at its start,
/// returning `true` when `node`'s *entire* language is that fixed run (so a
/// concatenation may keep going into the next node) and `false` when the
/// contribution stops here.
fn node_prefix(node: &Node, out: &mut Vec<u8>) -> bool {
    match node {
        Node::Char(c) => {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            true
        }
        Node::Str(s) => {
            out.extend_from_slice(s.as_bytes());
            true
        }
        Node::Concat(nodes) => {
            for n in nodes {
                if !node_prefix(n, out) {
                    return false;
                }
            }
            true
        }
        // `e{min,..}` with `min >= 1` forces `min` copies of whatever `e`
        // itself forces; anything past that is optional, so the run stops.
        Node::Repeat { inner, min, .. } if *min >= 1 => {
            let mut once = Vec::new();
            let exact = node_prefix(inner, &mut once);
            if once.is_empty() {
                return false;
            }
            for _ in 0..*min {
                out.extend_from_slice(&once);
                if !exact {
                    break;
                }
            }
            false
        }
        // `L(a & b)` is a subset of `L(a)` and of `L(b)`, so either side's
        // guaranteed prefix is guaranteed for the intersection; take the
        // longer of the two.
        Node::Intersect(a, b) => {
            let mut pa = Vec::new();
            node_prefix(a, &mut pa);
            let mut pb = Vec::new();
            node_prefix(b, &mut pb);
            out.extend_from_slice(if pb.len() > pa.len() { &pb } else { &pa });
            false
        }
        _ => false,
    }
}

/// Decodes the codepoint at the start of `term`, returning it with its UTF-8
/// width, or `None` when `term` is empty or does not begin with a well-formed
/// UTF-8 sequence. `None` means "no automaton compiled from a `RegExp` can
/// match here" -- real Lucene's `UTF32ToUTF8` only ever emits well-formed
/// UTF-8, so an ill-formed term byte is unmatchable there too.
fn next_char(term: &[u8]) -> Option<(u32, usize)> {
    let first = *term.first()?;
    let width = match first {
        0x00..=0x7F => return Some((first as u32, 1)),
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF4 => 4,
        _ => return None,
    };
    let bytes = term.get(..width)?;
    // `str::from_utf8` rejects overlongs, surrogates and out-of-range
    // sequences, which is exactly the set `UTF32ToUTF8` never produces.
    let s = std::str::from_utf8(bytes).ok()?;
    let c = s.chars().next()?;
    Some((c as u32, width))
}

/// Backtracking matcher: does `node` match some prefix of `term`, such that
/// `cont` (given the unconsumed remainder) also succeeds? Continuation-
/// passing lets `Concat`/`Alt`/`Intersect`/quantifiers compose correctly
/// across arbitrary nesting (a group's internal choice of how much to consume
/// must be allowed to depend on what comes *after* the group).
// ARITH: `remaining == 0` returns before the decrement, so `remaining - 1`
// cannot underflow. `term.len() - rest.len()` is safe because every `rest`
// handed to a continuation is a *suffix* of the `term` that produced it --
// the matcher only ever passes `&term[k..]` downward -- so its length is at
// most `term.len()`.
#[allow(clippy::arithmetic_side_effects)]
fn node_match(node: &Node, term: &[u8], budget: &Cell<u64>, cont: &dyn Fn(&[u8]) -> bool) -> bool {
    // Charge every node visited, not just quantifier iterations: nested
    // bounded repetition (`{n,m}`) can combine multiplicatively (see
    // `RegexpPattern::matches`'s doc comment), so the budget must cap total
    // backtracking work regardless of which node shape drives it.
    let remaining = budget.get();
    if remaining == 0 {
        return false;
    }
    budget.set(remaining - 1);
    match node {
        Node::Empty => false,
        Node::AnyChar => match next_char(term) {
            Some((_, w)) => cont(&term[w..]),
            None => false,
        },
        Node::Char(c) => match next_char(term) {
            Some((cp, w)) if cp == *c as u32 => cont(&term[w..]),
            _ => false,
        },
        Node::Str(s) => match term.strip_prefix(s.as_bytes()) {
            Some(rest) => cont(rest),
            None => false,
        },
        Node::Class { ranges, negated } => match next_char(term) {
            Some((cp, w)) => {
                let in_class = ranges.iter().any(|&(lo, hi)| cp >= lo && cp <= hi);
                // `[^...]` is `anyChar & complement(class)` in Lucene, so it
                // is still bounded above by `MAX_CODE_POINT`; `next_char`
                // already guarantees that, but stating it keeps the parallel
                // to `Automata.makeAnyChar` explicit.
                if in_class != *negated && cp <= MAX_CODE_POINT {
                    cont(&term[w..])
                } else {
                    false
                }
            }
            None => false,
        },
        Node::Concat(nodes) => concat_match(nodes, term, budget, cont),
        Node::Alt(alts) => alts.iter().any(|n| node_match(n, term, budget, cont)),
        Node::Intersect(a, b) => node_match(a, term, budget, &|rest| {
            // Both sides must accept exactly the same consumed span; only
            // then may the continuation run on what is left.
            let consumed = &term[..term.len() - rest.len()];
            node_match(b, consumed, budget, &|r| r.is_empty()) && cont(rest)
        }),
        Node::Repeat { inner, min, max } => repeat_match(inner, term, *min, *max, budget, cont),
        Node::Interval { min, max, digits } => {
            interval_match(term, *min, *max, *digits, budget, cont)
        }
    }
}

/// `inner{min,max}` against `term` (`max == None` means unbounded). Consumes
/// the mandatory `min` repetitions first (recursion terminates because `min`
/// strictly decreases each step, regardless of whether `inner` makes byte
/// progress), then tries the shortest match first and grows, refusing to
/// recurse when a repetition made no progress (which would otherwise loop
/// forever on a zero-width `inner`, e.g. `(a?)*`).
// ARITH: `min - 1` runs only under `min > 0`. `max.map(|m| m - 1)` needs
// `m > 0`: on the first call site that is `max >= min > 0`. `Node` is a
// private enum, and every one of the five places a `Node::Repeat` is built
// establishes `max.is_none() || max >= min` -- `?` (0,1), `*` (0,None), `+`
// (1,None), `@` (0,None), and `parse_repeat_bounds`, which rejects
// `min > max` with `InvalidRepetitionRange`. The recursion decrements both
// together, so the relation is preserved. On the second call site
// `max == Some(0)` has already returned, so `m >= 1`.
#[allow(clippy::arithmetic_side_effects)]
fn repeat_match(
    inner: &Node,
    term: &[u8],
    min: u32,
    max: Option<u32>,
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    if min > 0 {
        return node_match(inner, term, budget, &|rest| {
            repeat_match(inner, rest, min - 1, max.map(|m| m - 1), budget, cont)
        });
    }
    if max == Some(0) {
        return cont(term);
    }
    if cont(term) {
        return true;
    }
    node_match(inner, term, budget, &|rest| {
        if rest.len() == term.len() {
            false
        } else {
            repeat_match(inner, rest, 0, max.map(|m| m - 1), budget, cont)
        }
    })
}

/// `<min-max>` against `term` -- `Automata.makeDecimalInterval`'s language.
///
/// With `digits > 0` the match is exactly `digits` decimal digits wide,
/// zero-padded (Lucene's `between(..., zeros = false)` with no `0*` prefix
/// loop). With `digits == 0` any width is accepted, leading zeros included
/// (Lucene's extra initial state with a `'0'` self-loop and epsilons into
/// every all-zeros-so-far state); in both cases the decimal value the digits
/// spell must lie in `min..=max`.
// ARITH: `budget.get() == 0` returns before the decrement.
#[allow(clippy::arithmetic_side_effects)]
fn interval_match(
    term: &[u8],
    min: u32,
    max: u32,
    digits: usize,
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    let available = term.iter().take_while(|b| b.is_ascii_digit()).count();
    if digits > 0 {
        return available >= digits
            && decimal_value_in_range(&term[..digits], min, max)
            && cont(&term[digits..]);
    }
    for len in 1..=available {
        if budget.get() == 0 {
            return false;
        }
        budget.set(budget.get() - 1);
        if decimal_value_in_range(&term[..len], min, max) && cont(&term[len..]) {
            return true;
        }
    }
    false
}

/// Whether the ASCII decimal digits in `digits` spell a value in `min..=max`.
/// Leading zeros are insignificant; a value too large for the `u32` bounds is
/// simply out of range rather than an overflow.
// ARITH: the `significant.len() > 10` early return caps the accumulator at
// 10 decimal digits, so `value` stays below `10^10` and `value * 10 + d`
// below `10^11` -- three orders of magnitude inside `u64`. Every byte in
// `significant` is an ASCII digit (the callers slice a run
// `take_while(is_ascii_digit)` proved), so `b - b'0'` cannot underflow.
#[allow(clippy::arithmetic_side_effects)]
fn decimal_value_in_range(digits: &[u8], min: u32, max: u32) -> bool {
    let significant = digits
        .iter()
        .position(|b| *b != b'0')
        .map_or(&digits[digits.len()..], |i| &digits[i..]);
    // `u32::MAX` is 10 digits, so 11 significant digits cannot be in range.
    if significant.len() > 10 {
        return false;
    }
    let mut value: u64 = 0;
    for b in significant {
        value = value * 10 + u64::from(b - b'0');
    }
    value >= u64::from(min) && value <= u64::from(max)
}

/// Backtracking matcher in **prefix** mode: could *any* string beginning with
/// `term` match `node` (and then `cont`)?
///
/// Running out of input part-way through a node counts as success, because a
/// longer term could still supply what is missing. This is the "is this state
/// dead?" question `ByteRunAutomaton` answers for `IntersectTermsEnum` -- see
/// [`RegexpPattern::dead_prefix_len`] -- expressed for a backtracker instead
/// of a DFA.
///
/// It is a sound **over**-approximation: it may return `true` where nothing
/// actually matches (costing pruning, never correctness), and it never returns
/// `false` when some extension could match. The one place that matters is
/// `Node::Intersect`, where only the left side is consulted, since
/// `L(a & b)` is a subset of `L(a)`.
// ARITH: same two invariants as `node_match` -- the budget is checked against
// zero before it is decremented, and every `rest` is a suffix of its `term`.
#[allow(clippy::arithmetic_side_effects)]
fn node_prefix_match(
    node: &Node,
    term: &[u8],
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    let remaining = budget.get();
    if remaining == 0 {
        // Out of budget: claim viability rather than pruning, so a pattern
        // that exhausts the budget degrades to a full scan rather than to
        // wrong answers.
        return true;
    }
    budget.set(remaining - 1);
    match node {
        Node::Empty => false,
        Node::AnyChar => match next_char(term) {
            Some((_, w)) => cont(&term[w..]),
            // Empty input: a longer term supplies the character.
            None => term.is_empty(),
        },
        Node::Char(c) => match next_char(term) {
            Some((cp, w)) => cp == *c as u32 && cont(&term[w..]),
            None => term.is_empty(),
        },
        Node::Class { ranges, negated } => match next_char(term) {
            Some((cp, w)) => {
                let in_class = ranges.iter().any(|&(lo, hi)| cp >= lo && cp <= hi);
                in_class != *negated && cp <= MAX_CODE_POINT && cont(&term[w..])
            }
            None => term.is_empty(),
        },
        Node::Str(s) => {
            if let Some(rest) = term.strip_prefix(s.as_bytes()) {
                cont(rest)
            } else {
                // A term that is itself a prefix of the literal is still
                // viable.
                s.as_bytes().starts_with(term)
            }
        }
        Node::Concat(nodes) => concat_prefix_match(nodes, term, budget, cont),
        Node::Alt(alts) => alts
            .iter()
            .any(|n| node_prefix_match(n, term, budget, cont)),
        Node::Intersect(a, _) => node_prefix_match(a, term, budget, cont),
        Node::Repeat { inner, min, max } => {
            repeat_prefix_match(inner, term, *min, *max, budget, cont)
        }
        Node::Interval { min, max, digits } => {
            let available = term.iter().take_while(|b| b.is_ascii_digit()).count();
            // A digit run that reaches the end of the input can still grow
            // into (or past) a matching one, so it stays viable. Everything
            // else is decided by the real interval match, with the
            // prefix-mode continuation carrying on from there.
            if available == term.len() {
                return true;
            }
            interval_match(term, *min, *max, *digits, budget, cont)
        }
    }
}

fn concat_prefix_match(
    nodes: &[Node],
    term: &[u8],
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    match nodes.split_first() {
        None => cont(term),
        Some((first, rest)) => node_prefix_match(first, term, budget, &|r| {
            concat_prefix_match(rest, r, budget, cont)
        }),
    }
}

/// [`repeat_match`] in prefix mode.
// ARITH: identical to `repeat_match`'s: `min > 0` guards `min - 1`, and
// `max.is_none() || max >= min` (established by all five `Node::Repeat`
// construction sites, preserved by the paired decrements) plus the
// `max == Some(0)` early return guard `m - 1`.
#[allow(clippy::arithmetic_side_effects)]
fn repeat_prefix_match(
    inner: &Node,
    term: &[u8],
    min: u32,
    max: Option<u32>,
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    if term.is_empty() {
        // Anything still to come can be supplied by a longer term, except
        // where the body itself accepts nothing.
        return !matches!(inner, Node::Empty) || min == 0;
    }
    if min > 0 {
        return node_prefix_match(inner, term, budget, &|rest| {
            repeat_prefix_match(inner, rest, min - 1, max.map(|m| m - 1), budget, cont)
        });
    }
    if max == Some(0) {
        return cont(term);
    }
    if cont(term) {
        return true;
    }
    node_prefix_match(inner, term, budget, &|rest| {
        if rest.len() == term.len() {
            false
        } else {
            repeat_prefix_match(inner, rest, 0, max.map(|m| m - 1), budget, cont)
        }
    })
}

fn concat_match(
    nodes: &[Node],
    term: &[u8],
    budget: &Cell<u64>,
    cont: &dyn Fn(&[u8]) -> bool,
) -> bool {
    match nodes.split_first() {
        None => cont(term),
        Some((first, rest)) => node_match(first, term, budget, &|r| {
            concat_match(rest, r, budget, cont)
        }),
    }
}

/// Recursive-descent parser mirroring `RegExp`'s own productions
/// (`parseUnionExp` / `parseInterExp` / `parseConcatExp` / `parseRepeatExp` /
/// `parseCharClassExp` / `parseCharClasses` / `parseSimpleExp` /
/// `parseCharExp`), one method each, with `RegExp.ALL` syntax flags always
/// on. Positions are codepoint indices, matching `RegExp`'s error messages
/// for a BMP-only pattern.
struct Parser<'a> {
    chars: &'a [char],
    pos: usize,
}

impl Parser<'_> {
    fn peek_is(&self, set: &str) -> bool {
        self.chars.get(self.pos).is_some_and(|c| set.contains(*c))
    }

    fn more(&self) -> bool {
        self.pos < self.chars.len()
    }

    /// `RegExp.match(c)`: consume `c` if it is next.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn eat(&mut self, c: char) -> bool {
        if self.chars.get(self.pos) == Some(&c) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    /// `RegExp.next()`: consume and return the next codepoint, or fail.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Result<char, RegexpError> {
        let c = *self
            .chars
            .get(self.pos)
            .ok_or(RegexpError::UnexpectedEndOfString)?;
        self.pos += 1;
        Ok(c)
    }

    /// `parseUnionExp`: `inter ('|' inter)*`.
    fn parse_union(&mut self) -> Result<Node, RegexpError> {
        let mut alts = vec![self.parse_inter()?];
        while self.eat('|') {
            alts.push(self.parse_inter()?);
        }
        Ok(if alts.len() == 1 {
            alts.pop().expect("just checked len == 1")
        } else {
            Node::Alt(alts)
        })
    }

    /// `parseInterExp`: `concat ('&' concat)*` (the `INTERSECTION` flag is
    /// always on here, since `RegexpQuery` uses `RegExp.ALL`).
    fn parse_inter(&mut self) -> Result<Node, RegexpError> {
        let mut node = self.parse_concat()?;
        while self.eat('&') {
            let rhs = self.parse_concat()?;
            node = Node::Intersect(Box::new(node), Box::new(rhs));
        }
        Ok(node)
    }

    /// `parseConcatExp`: one atom, then more until `)`, `|`, `&` or end of
    /// input. The first atom is gathered **unconditionally** -- Lucene's
    /// `iterativeParseExp` calls its gather function before testing its stop
    /// condition -- so a pattern that opens with `|` or `&` takes that
    /// codepoint as an ordinary literal rather than as an operator with an
    /// empty left-hand side.
    fn parse_concat(&mut self) -> Result<Node, RegexpError> {
        let mut nodes = vec![self.parse_repeat()?];
        while self.more() && !self.peek_is(")|&") {
            nodes.push(self.parse_repeat()?);
        }
        Ok(if nodes.len() == 1 {
            nodes.pop().expect("just checked len == 1")
        } else {
            Node::Concat(nodes)
        })
    }

    /// `parseRepeatExp`: an atom followed by **any number** of postfix
    /// quantifiers -- Lucene loops `while (peek("?*+{"))`, so `a**` and
    /// `a?+` are legal (and mean the same as `a*`), not parse errors.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn parse_repeat(&mut self) -> Result<Node, RegexpError> {
        let mut node = self.parse_char_class_exp()?;
        while self.peek_is("?*+{") {
            if self.eat('?') {
                node = Node::Repeat {
                    inner: Box::new(node),
                    min: 0,
                    max: Some(1),
                };
            } else if self.eat('*') {
                node = Node::Repeat {
                    inner: Box::new(node),
                    min: 0,
                    max: None,
                };
            } else if self.eat('+') {
                node = Node::Repeat {
                    inner: Box::new(node),
                    min: 1,
                    max: None,
                };
            } else {
                // `{`
                self.pos += 1;
                node = self.parse_repeat_bounds(node)?;
            }
        }
        Ok(node)
    }

    /// The `{n}` / `{n,}` / `{n,m}` tail of `parseRepeatExp`, entered with the
    /// `{` already consumed.
    fn parse_repeat_bounds(&mut self, inner: Node) -> Result<Node, RegexpError> {
        let digits_at = self.pos;
        let min = self
            .parse_number()?
            .ok_or(RegexpError::IntegerExpected { pos: digits_at })?;
        let max = if self.eat(',') {
            self.parse_number()?
        } else {
            Some(min)
        };
        if !self.eat('}') {
            return Err(RegexpError::ExpectedChar {
                expected: '}',
                pos: self.pos,
            });
        }
        if let Some(max) = max {
            if min > max {
                return Err(RegexpError::InvalidRepetitionRange { min, max });
            }
        }
        Ok(Node::Repeat {
            inner: Box::new(inner),
            min,
            max,
        })
    }

    /// A run of ASCII decimal digits, or `None` when there are none here.
    /// Java parses with `Integer.parseInt`, which throws on overflow; this
    /// reports the same failure as [`RegexpError::IntegerExpected`] since
    /// there is no other channel for it and no legitimate pattern reaches it.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn parse_number(&mut self) -> Result<Option<u32>, RegexpError> {
        let start = self.pos;
        while self.peek_is("0123456789") {
            self.pos += 1;
        }
        if self.pos == start {
            return Ok(None);
        }
        let text: String = self.chars[start..self.pos].iter().collect();
        text.parse::<u32>()
            .map(Some)
            .map_err(|_| RegexpError::IntegerExpected { pos: start })
    }

    /// `parseCharClassExp`: `'[' '^'? charclasses ']'` or a simple atom.
    fn parse_char_class_exp(&mut self) -> Result<Node, RegexpError> {
        if self.eat('[') {
            let negated = self.eat('^');
            let ranges = self.parse_char_classes()?;
            if !self.eat(']') {
                return Err(RegexpError::ExpectedChar {
                    expected: ']',
                    pos: self.pos,
                });
            }
            Ok(Node::Class { ranges, negated })
        } else {
            self.parse_simple()
        }
    }

    /// `parseCharClasses`: a do-while over class items, so `[]` is not an
    /// "empty class" error but a `]`-as-a-member parse that then fails to
    /// find its closing bracket -- exactly Lucene's behaviour.
    ///
    /// An **escaped** item (`\x`) is a single codepoint and may not open a
    /// range: Lucene's escape branch adds `c` and loops without ever checking
    /// for a following `-`, so `[\--z]` is the three members `-`, `-`, `z`,
    /// not the range `-`..`z`.
    fn parse_char_classes(&mut self) -> Result<Vec<(u32, u32)>, RegexpError> {
        let mut ranges = Vec::new();
        loop {
            if self.eat('\\') {
                if self.peek_is(PREDEFINED_ESCAPE_LEADS) {
                    self.expand_predefined(&mut ranges)?;
                } else {
                    let c = self.next()? as u32;
                    ranges.push((c, c));
                }
            } else {
                let lo = self.parse_char_exp()? as u32;
                if self.eat('-') {
                    let hi = self.parse_char_exp()? as u32;
                    ranges.push((lo, hi));
                } else {
                    ranges.push((lo, lo));
                }
            }
            if !(self.more() && !self.peek_is("]")) {
                break;
            }
        }
        Ok(ranges)
    }

    /// `expandPreDefined`: the `\\`, `\d`, `\D`, `\s`, `\S`, `\w`, `\W`
    /// classes, with every other alphabetic escape an error. Entered with the
    /// `\` already consumed and an alphabetic (or `\`) codepoint next.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    // The `- 1`/`+ 1` on the class boundaries are on ASCII constants
    // (`\t`, `\n`, `\r`, ' ', '0', '9', 'A', 'Z', '_', 'a', 'z'), none of
    // which is 0 or `u32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn expand_predefined(&mut self, ranges: &mut Vec<(u32, u32)>) -> Result<(), RegexpError> {
        let c = *self.chars.get(self.pos).expect("caller checked peek");
        self.pos += 1;
        match c {
            '\\' => ranges.push(('\\' as u32, '\\' as u32)),
            'd' => ranges.push((u32::from(b'0'), u32::from(b'9'))),
            'D' => {
                ranges.push((0, u32::from(b'0') - 1));
                ranges.push((u32::from(b'9') + 1, MAX_CODE_POINT));
            }
            's' => {
                ranges.push((u32::from(b'\t'), u32::from(b'\n')));
                ranges.push((u32::from(b'\r'), u32::from(b'\r')));
                ranges.push((u32::from(b' '), u32::from(b' ')));
            }
            'S' => {
                ranges.push((0, u32::from(b'\t') - 1));
                ranges.push((u32::from(b'\n') + 1, u32::from(b'\r') - 1));
                ranges.push((u32::from(b'\r') + 1, u32::from(b' ') - 1));
                ranges.push((u32::from(b' ') + 1, MAX_CODE_POINT));
            }
            'w' => {
                ranges.push((u32::from(b'0'), u32::from(b'9')));
                ranges.push((u32::from(b'A'), u32::from(b'Z')));
                ranges.push((u32::from(b'_'), u32::from(b'_')));
                ranges.push((u32::from(b'a'), u32::from(b'z')));
            }
            'W' => {
                ranges.push((0, u32::from(b'0') - 1));
                ranges.push((u32::from(b'9') + 1, u32::from(b'A') - 1));
                ranges.push((u32::from(b'Z') + 1, u32::from(b'_') - 1));
                ranges.push((u32::from(b'_') + 1, u32::from(b'a') - 1));
                ranges.push((u32::from(b'z') + 1, MAX_CODE_POINT));
            }
            other => return Err(RegexpError::InvalidCharacterClass(other)),
        }
        Ok(())
    }

    /// `matchPredefinedCharacterClass`: a `\` followed by an alphabetic
    /// codepoint (or another `\`) outside a `[...]`. Returns `None` -- with
    /// the `\` consumed, exactly as Lucene's short-circuit leaves it -- when
    /// what follows is not one of those.
    fn match_predefined(&mut self) -> Result<Option<Node>, RegexpError> {
        if self.eat('\\') && self.peek_is(PREDEFINED_ESCAPE_LEADS) {
            let mut ranges = Vec::new();
            self.expand_predefined(&mut ranges)?;
            return Ok(Some(Node::Class {
                ranges,
                negated: false,
            }));
        }
        Ok(None)
    }

    /// `parseSimpleExp`.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    // The `self.pos - 1` inside is reached only after an `eat` that returned
    // true, which advanced `pos` past at least that one codepoint.
    #[allow(clippy::arithmetic_side_effects)]
    fn parse_simple(&mut self) -> Result<Node, RegexpError> {
        if self.eat('.') {
            return Ok(Node::AnyChar);
        }
        if self.eat('#') {
            return Ok(Node::Empty);
        }
        if self.eat('@') {
            // `makeAnyString()` -- any sequence of codepoints.
            return Ok(Node::Repeat {
                inner: Box::new(Node::AnyChar),
                min: 0,
                max: None,
            });
        }
        if self.eat('"') {
            let start = self.pos;
            while self.more() && !self.peek_is("\"") {
                self.pos += 1;
            }
            if !self.eat('"') {
                return Err(RegexpError::ExpectedChar {
                    expected: '"',
                    pos: self.pos,
                });
            }
            return Ok(Node::Str(self.chars[start..self.pos - 1].iter().collect()));
        }
        if self.eat('(') {
            if self.eat(')') {
                return Ok(Node::Str(String::new()));
            }
            let inner = self.parse_union()?;
            if !self.eat(')') {
                return Err(RegexpError::ExpectedChar {
                    expected: ')',
                    pos: self.pos,
                });
            }
            return Ok(inner);
        }
        if self.eat('<') {
            return self.parse_angle();
        }
        if let Some(node) = self.match_predefined()? {
            return Ok(node);
        }
        Ok(Node::Char(self.parse_char_exp()?))
    }

    /// The `<identifier>` / `<n-m>` production of `parseSimpleExp`, entered
    /// with the `<` already consumed.
    // ARITH: `self.pos` is a cursor into `self.chars`, only ever advanced past
    // a codepoint the surrounding `get`/`peek_is`/`more` proved is there, so
    // it never exceeds `self.chars.len()` -- itself at most `isize::MAX`.
    // `self.pos - 1` is reached only after `eat('>')` returned true. `s` is
    // non-empty whenever `s.len() - 1` runs, because `s.find('-')` returning
    // `None` returns first; `i + 1 <= s.len()` for the same reason; and
    // `s.len() - 1 - i` runs only past the `i == s.len() - 1` rejection, so
    // `i < s.len() - 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn parse_angle(&mut self) -> Result<Node, RegexpError> {
        let start = self.pos;
        while self.more() && !self.peek_is(">") {
            self.pos += 1;
        }
        if !self.eat('>') {
            return Err(RegexpError::ExpectedChar {
                expected: '>',
                pos: self.pos,
            });
        }
        let s: String = self.chars[start..self.pos - 1].iter().collect();
        let Some(i) = s.find('-') else {
            // AUTOMATON: no provider here, so this is Lucene's
            // `RegexpQuery.DEFAULT_PROVIDER` path, which always fails.
            return Err(RegexpError::NamedAutomatonNotFound(s));
        };
        let err = RegexpError::IntervalSyntax { pos: self.pos - 1 };
        if i == 0 || i == s.len() - 1 || Some(i) != s.rfind('-') {
            return Err(err);
        }
        let (Ok(a), Ok(b)) = (s[..i].parse::<u32>(), s[i + 1..].parse::<u32>()) else {
            return Err(err);
        };
        // `digits > 0` (a fixed, zero-padded width) only when both bounds
        // were written with the same number of characters.
        let digits = if i == s.len() - 1 - i { i } else { 0 };
        Ok(Node::Interval {
            min: a.min(b),
            max: a.max(b),
            digits,
        })
    }

    /// `parseCharExp`: an optional `\` then one codepoint.
    fn parse_char_exp(&mut self) -> Result<char, RegexpError> {
        self.eat('\\');
        self.next()
    }
}

/// The codepoints `RegExp` treats as "special escape or invalid escape" after
/// a `\`: every ASCII letter, plus `\` itself.
const PREDEFINED_ESCAPE_LEADS: &str = "\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

#[cfg(test)]
mod tests {
    use super::*;

    fn m(pattern: &str, term: &str) -> bool {
        RegexpPattern::parse(pattern)
            .unwrap()
            .matches(term.as_bytes())
    }

    fn err(pattern: &str) -> RegexpError {
        RegexpPattern::parse(pattern).unwrap_err()
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
        assert!(!m("ca", "cat"));
        assert!(m("ca", "ca"));
        assert!(!m("at", "cat"));
    }

    #[test]
    fn dot_matches_any_single_codepoint() {
        assert!(m("c.t", "cat"));
        assert!(m("c.t", "cot"));
        assert!(!m("c.t", "ct"));
        assert!(!m("c.t", "caat"));
        // `Automata.makeAnyChar` is one *codepoint*, not one UTF-8 byte:
        // `€` is 3 bytes and still matches a single `.`.
        assert!(m(".", "€"));
        assert!(m("a.c", "a€c"));
        assert!(m(".", "\u{10348}"));
    }

    #[test]
    fn ill_formed_utf8_matches_nothing() {
        // `UTF32ToUTF8` only ever emits well-formed UTF-8, so no compiled
        // `RegExp` automaton accepts an ill-formed term.
        let p = RegexpPattern::parse(".*").unwrap();
        assert!(p.matches(b"ok"));
        assert!(!p.matches(&[0xFF]));
        assert!(!p.matches(&[0x80]));
        // Truncated 3-byte sequence.
        assert!(!p.matches(&[0xE2, 0x82]));
        // Overlong encoding of '/'.
        assert!(!p.matches(&[0xC0, 0xAF]));
        // Surrogate half.
        assert!(!p.matches(&[0xED, 0xA0, 0x80]));
        assert!(!RegexpPattern::parse("[^a]").unwrap().matches(&[0xFF]));
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
    fn repeated_quantifiers_stack_like_lucene_rather_than_erroring() {
        // `parseRepeatExp` loops `while (peek("?*+{"))`, so `a**` is a repeat
        // of a repeat, not a parse error.
        assert!(m("a**", ""));
        assert!(m("a**", "aaa"));
        assert!(m("a?+", "aa"));
        assert!(m("a{2}?", ""));
        assert!(m("a{2}?", "aa"));
        assert!(!m("a{2}?", "a"));
    }

    #[test]
    fn character_class_matches_any_listed_codepoint() {
        assert!(m("[cb]at", "cat"));
        assert!(m("[cb]at", "bat"));
        assert!(!m("[cb]at", "hat"));
    }

    #[test]
    fn character_class_range_matches_any_codepoint_in_range() {
        assert!(m("[a-c]at", "aat"));
        assert!(m("[a-c]at", "bat"));
        assert!(m("[a-c]at", "cat"));
        assert!(!m("[a-c]at", "dat"));
    }

    #[test]
    fn negated_character_class_is_one_codepoint_not_listed() {
        assert!(m("[^ab]at", "cat"));
        assert!(!m("[^ab]at", "aat"));
        assert!(!m("[^ab]at", "bat"));
        // Lucene builds `[^..]` as `anyChar & complement(class)`, so it spans
        // the whole codepoint space, not just ASCII.
        assert!(m("[^a]", "€"));
        assert!(m("[^a]", "\u{10348}"));
    }

    #[test]
    fn escaped_member_inside_a_class_cannot_open_a_range() {
        // Lucene's escape branch in `parseCharClasses` adds the codepoint and
        // loops without checking for `-`, so `[\--z]` is the members `-`, `-`
        // and `z`, not the range `-`..`z`.
        assert!(m(r"[\--z]", "-"));
        assert!(m(r"[\--z]", "z"));
        assert!(!m(r"[\--z]", "a"));
    }

    #[test]
    fn dash_before_the_closing_bracket_is_a_range_end_in_lucene() {
        // `[a-]` is *not* "a literal trailing dash": `parseCharExp` consumes
        // the `]` as the range's upper bound, and the class is then unclosed.
        assert_eq!(
            err("[a-]"),
            RegexpError::ExpectedChar {
                expected: ']',
                pos: 4
            }
        );
    }

    #[test]
    fn escaped_bracket_inside_class_is_a_literal_class_member() {
        assert!(m(r"[\]]", "]"));
        assert!(!m(r"[\]]", "x"));
    }

    #[test]
    fn trailing_backslash_inside_class_is_an_end_of_string_error() {
        assert_eq!(err(r"[a\"), RegexpError::UnexpectedEndOfString);
    }

    #[test]
    fn predefined_classes_expand_like_javas() {
        assert!(m(r"\d+", "1234"));
        assert!(!m(r"\d+", "12a4"));
        assert!(m(r"\D", "a"));
        assert!(!m(r"\D", "1"));
        assert!(m(r"\w+", "a_Z9"));
        assert!(!m(r"\w+", "a-b"));
        assert!(m(r"\W", "-"));
        assert!(m(r"\s", " "));
        assert!(m(r"\s", "\t"));
        assert!(m(r"\s", "\n"));
        assert!(m(r"\s", "\r"));
        assert!(!m(r"\s", "a"));
        assert!(m(r"\S", "a"));
        assert!(!m(r"\S", " "));
        // ... and inside a class, where they union with the other members.
        assert!(m(r"[\dx]+", "1x2"));
        assert!(!m(r"[\dx]+", "1y2"));
        assert!(m(r"[^\d]", "a"));
        assert!(!m(r"[^\d]", "7"));
        // `\\` is the escape for a literal backslash in both positions.
        assert!(m(r"\\", "\\"));
        assert!(m(r"[\\]", "\\"));
    }

    #[test]
    fn an_alphabetic_escape_that_is_not_a_predefined_class_is_an_error() {
        assert_eq!(err(r"\A"), RegexpError::InvalidCharacterClass('A'));
        assert_eq!(err(r"\q"), RegexpError::InvalidCharacterClass('q'));
        assert_eq!(err(r"[\A]"), RegexpError::InvalidCharacterClass('A'));
    }

    #[test]
    fn alternation_matches_either_side() {
        assert!(m("cat|dog", "cat"));
        assert!(m("cat|dog", "dog"));
        assert!(!m("cat|dog", "bird"));
    }

    #[test]
    fn intersection_requires_both_sides_to_accept_the_same_string() {
        // `&` is enabled by `RegExp.ALL`, which is what `RegexpQuery(Term)`
        // uses. It used to be rejected outright by this port.
        assert!(m("[a-z]+&...", "abc"));
        assert!(!m("[a-z]+&...", "ab"));
        assert!(!m("[a-z]+&...", "ab1"));
        assert!(m("(cat|dog)&(dog|bird)", "dog"));
        assert!(!m("(cat|dog)&(dog|bird)", "cat"));
        // Intersection binds tighter than union.
        assert!(m("ab&ab|cd", "cd"));
        assert!(m("ab&ab|cd", "ab"));
        // ... and looser than concatenation, so both sides are whole spans.
        assert!(m("a.&.b", "ab"));
        assert!(!m("a.&.b", "ac"));
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
    fn empty_group_is_the_empty_string() {
        assert!(m("()", ""));
        assert!(!m("()", "a"));
        assert!(m("a()b", "ab"));
    }

    #[test]
    fn quoted_string_is_a_literal_with_no_escapes_inside() {
        assert!(m(r#""a*b""#, "a*b"));
        assert!(!m(r#""a*b""#, "aab"));
        assert!(m(r#""""#, ""));
        assert!(m(r#"x"[](){}"y"#, "x[](){}y"));
        assert_eq!(
            err(r#""abc"#),
            RegexpError::ExpectedChar {
                expected: '"',
                pos: 4
            }
        );
    }

    #[test]
    fn hash_is_the_empty_language() {
        // `#` (the `EMPTY` flag, on in `RegExp.ALL`) accepts nothing at all,
        // not the empty string.
        assert!(!m("#", ""));
        assert!(!m("#", "a"));
        assert!(!m("a#b", "ab"));
        // Union with the empty language is the other side.
        assert!(m("#|cat", "cat"));
    }

    #[test]
    fn at_sign_is_any_string() {
        assert!(m("@", ""));
        assert!(m("@", "anything at all"));
        assert!(m("@", "€uro"));
        assert!(m("cat@", "catalog"));
        assert!(!m("cat@", "dog"));
    }

    #[test]
    fn tilde_is_an_ordinary_literal_because_complement_is_not_in_regexp_all() {
        // `DEPRECATED_COMPLEMENT` (0x10000) is not part of `RegExp.ALL`, so
        // `RegexpQuery(Term)` treats `~` as a plain character. This port used
        // to reject the whole pattern.
        assert!(m("a~b", "a~b"));
        assert!(!m("a~b", "ab"));
        assert!(m("~", "~"));
    }

    #[test]
    fn numeric_interval_with_equal_width_bounds_is_fixed_width() {
        // `<05-40>`: both bounds two characters, so `digits == 2` and only
        // zero-padded two-digit strings match.
        assert!(m("<05-40>", "05"));
        assert!(m("<05-40>", "40"));
        assert!(m("<05-40>", "23"));
        assert!(!m("<05-40>", "5"));
        assert!(!m("<05-40>", "41"));
        assert!(!m("<05-40>", "004"));
    }

    #[test]
    fn numeric_interval_with_unequal_width_bounds_accepts_any_width() {
        // `<5-40>`: bounds of different widths, so `digits == 0` and Lucene
        // adds a `0*` prefix loop -- any number of leading zeros, any width.
        assert!(m("<5-40>", "5"));
        assert!(m("<5-40>", "05"));
        assert!(m("<5-40>", "0005"));
        assert!(m("<5-40>", "40"));
        assert!(!m("<5-40>", "4"));
        assert!(!m("<5-40>", "41"));
        assert!(!m("<5-40>", ""));
    }

    #[test]
    fn numeric_interval_swaps_out_of_order_bounds_and_composes() {
        assert!(m("<40-5>", "23"));
        assert!(m("x<1-3>y", "x2y"));
        assert!(!m("x<1-3>y", "x4y"));
        // A value far beyond `u32` is out of range, not an overflow panic.
        assert!(!m("<1-30>", "999999999999999999999"));
        assert!(m("<1-30>", "0000000009"));
    }

    #[test]
    fn malformed_interval_is_a_parse_error() {
        assert_eq!(err("<-3>"), RegexpError::IntervalSyntax { pos: 3 });
        assert_eq!(err("<1->"), RegexpError::IntervalSyntax { pos: 3 });
        assert_eq!(err("<1-2-3>"), RegexpError::IntervalSyntax { pos: 6 });
        assert_eq!(err("<a-b>"), RegexpError::IntervalSyntax { pos: 4 });
        assert_eq!(
            err("<1-2"),
            RegexpError::ExpectedChar {
                expected: '>',
                pos: 4
            }
        );
    }

    #[test]
    fn named_automaton_is_not_found_because_there_is_no_provider() {
        // `RegexpQuery.DEFAULT_PROVIDER` returns null for every name, so
        // `RegExp.toAutomaton` throws `"'name' not found"`.
        assert_eq!(
            err("<name>"),
            RegexpError::NamedAutomatonNotFound("name".to_string())
        );
    }

    #[test]
    fn nested_groups_and_alternation_compose() {
        assert!(m("(a(b|c)d)+", "abdacd"));
        assert!(!m("(a(b|c)d)+", "abdaed"));
    }

    #[test]
    fn escaped_metacharacter_is_a_literal() {
        assert!(m(r"a\*b", "a*b"));
        assert!(!m(r"a\*b", "aab"));
        assert!(m(r"a\.b", "a.b"));
        assert!(!m(r"a\.b", "axb"));
        assert!(m(r"a\[b", "a[b"));
        assert!(m(r"a\&b", "a&b"));
        assert!(m(r"a\@b", "a@b"));
        assert!(m(r"a\#b", "a#b"));
    }

    #[test]
    fn trailing_unescaped_backslash_is_an_end_of_string_error() {
        // Lucene's `parseCharExp` calls `next()` past the end and throws.
        assert_eq!(err(r"ab\"), RegexpError::UnexpectedEndOfString);
    }

    #[test]
    fn empty_pattern_matches_only_empty_term() {
        assert!(m("", ""));
        assert!(!m("", "a"));
    }

    #[test]
    fn a_leading_quantifier_is_an_ordinary_literal_in_lucene() {
        // Lucene's `parseSimpleExp` falls through to `makeChar`, so `*cat`
        // matches the literal term `*cat` -- it is not a "dangling
        // quantifier" error the way this port used to report.
        assert!(m("*cat", "*cat"));
        assert!(m("{2,3}", "{2,3}"));
        assert!(m("+x", "+x"));
        assert!(m("?", "?"));
    }

    #[test]
    fn a_leading_union_or_intersection_operator_is_a_literal() {
        // `iterativeParseExp` gathers before testing its stop condition, so
        // the operator at position 0 has no left-hand side and falls through
        // to `makeChar`.
        assert!(m("|a", "|a"));
        assert!(m("&a", "&a"));
        assert!(!m("|a", "a"));
    }

    #[test]
    fn a_trailing_operator_runs_off_the_end_of_the_pattern() {
        assert_eq!(err("a|"), RegexpError::UnexpectedEndOfString);
        assert_eq!(err("a&"), RegexpError::UnexpectedEndOfString);
    }

    #[test]
    fn unmatched_open_paren_is_a_parse_error() {
        assert_eq!(
            err("(cat"),
            RegexpError::ExpectedChar {
                expected: ')',
                pos: 4
            }
        );
    }

    #[test]
    fn unmatched_close_paren_is_a_parse_error() {
        assert_eq!(err("cat)"), RegexpError::EndOfStringExpected { pos: 3 });
    }

    #[test]
    fn unmatched_open_bracket_is_a_parse_error() {
        assert_eq!(
            err("[cat"),
            RegexpError::ExpectedChar {
                expected: ']',
                pos: 4
            }
        );
    }

    #[test]
    fn empty_class_is_an_unclosed_class_error_not_an_empty_class_error() {
        // Lucene's do-while consumes the `]` as a member, then fails to find
        // the real closing bracket.
        assert_eq!(
            err("[]"),
            RegexpError::ExpectedChar {
                expected: ']',
                pos: 2
            }
        );
        assert_eq!(
            err("[^]"),
            RegexpError::ExpectedChar {
                expected: ']',
                pos: 3
            }
        );
    }

    #[test]
    fn every_error_variant_has_a_non_empty_display_message() {
        for e in [
            RegexpError::UnexpectedEndOfString,
            RegexpError::ExpectedChar {
                expected: ')',
                pos: 1,
            },
            RegexpError::IntegerExpected { pos: 1 },
            RegexpError::InvalidRepetitionRange { min: 3, max: 2 },
            RegexpError::InvalidCharacterClass('A'),
            RegexpError::IntervalSyntax { pos: 1 },
            RegexpError::IllegalIdentifier { pos: 1 },
            RegexpError::NamedAutomatonNotFound("x".to_string()),
            RegexpError::EndOfStringExpected { pos: 1 },
        ] {
            assert!(!e.to_string().is_empty());
        }
        assert_eq!(
            RegexpError::InvalidRepetitionRange { min: 3, max: 2 }.to_string(),
            "invalid repetition range(out of order): 3..2"
        );
    }

    // -- {n,m} bounded repetition -------------------------------------

    #[test]
    fn exact_count_repeat_matches_only_that_many() {
        assert!(!m("a{3}", "aa"));
        assert!(m("a{3}", "aaa"));
        assert!(!m("a{3}", "aaaa"));
    }

    #[test]
    fn zero_zero_repeat_matches_zero_occurrences_only() {
        assert!(m("a{0,0}b", "b"));
        assert!(!m("a{0,0}b", "ab"));
    }

    #[test]
    fn min_only_repeat_is_unbounded_above() {
        assert!(!m("a{2,}", "a"));
        assert!(m("a{2,}", "aa"));
        assert!(m("a{2,}", "aaa"));
        assert!(m("a{2,}", "aaaaaaaa"));
    }

    #[test]
    fn min_max_repeat_bounds_both_ends() {
        assert!(!m("a{2,4}", "a"));
        assert!(m("a{2,4}", "aa"));
        assert!(m("a{2,4}", "aaa"));
        assert!(m("a{2,4}", "aaaa"));
        assert!(!m("a{2,4}", "aaaaa"));
    }

    #[test]
    fn repeat_zero_min_allows_absence() {
        assert!(m("a{0,2}b", "b"));
        assert!(m("a{0,2}b", "ab"));
        assert!(m("a{0,2}b", "aab"));
        assert!(!m("a{0,2}b", "aaab"));
    }

    #[test]
    fn repeat_composes_with_other_operators() {
        assert!(m("a{2,3}b*", "aa"));
        assert!(m("a{2,3}b*", "aaab"));
        assert!(m("a{2,3}b*", "aaabbb"));
        assert!(!m("a{2,3}b*", "a"));
        assert!(!m("a{2,3}b*", "aaaab"));
    }

    #[test]
    fn repeat_on_group_applies_to_whole_group() {
        assert!(m("(ab){2,3}", "abab"));
        assert!(m("(ab){2,3}", "ababab"));
        assert!(!m("(ab){2,3}", "ab"));
        assert!(!m("(ab){2,3}", "abababab"));
    }

    /// A nested bounded-repeat pattern whose two `{1,15}` counts combine
    /// multiplicatively against a matching-but-ultimately-failing input
    /// (an all-`a` term with no trailing `b`) -- without the step budget in
    /// `RegexpPattern::matches`, this would backtrack combinatorially and
    /// hang; with it, this must return promptly (the test itself times out
    /// the whole suite otherwise) and correctly report no match.
    #[test]
    fn nested_bounded_repeat_does_not_hang_on_a_failing_match() {
        assert!(!m("(a{1,15}){1,15}b", &"a".repeat(40)));
    }

    #[test]
    fn zero_width_repeat_inner_terminates() {
        // `(a?)*` can consume nothing per iteration; without the
        // no-progress guard in `repeat_match` this loops forever.
        assert!(m("(a?)*", "aaa"));
        assert!(m("(a?)*", ""));
        assert!(!m("(a?)*b", "aaa"));
    }

    #[test]
    fn malformed_repeat_missing_close_brace_is_a_parse_error() {
        assert_eq!(
            err("a{2,3"),
            RegexpError::ExpectedChar {
                expected: '}',
                pos: 5
            }
        );
    }

    #[test]
    fn malformed_repeat_non_numeric_bound_is_a_parse_error() {
        assert_eq!(err("a{x}"), RegexpError::IntegerExpected { pos: 2 });
        assert_eq!(err("a{}"), RegexpError::IntegerExpected { pos: 2 });
    }

    #[test]
    fn malformed_repeat_max_less_than_min_is_a_parse_error() {
        assert_eq!(
            err("a{3,2}"),
            RegexpError::InvalidRepetitionRange { min: 3, max: 2 }
        );
    }

    #[test]
    fn repeat_bound_too_large_for_u32_is_a_parse_error() {
        assert_eq!(
            err("a{99999999999}"),
            RegexpError::IntegerExpected { pos: 2 }
        );
    }

    // -- literal_prefix ------------------------------------------------

    fn prefix(pattern: &str) -> Vec<u8> {
        RegexpPattern::parse(pattern).unwrap().literal_prefix()
    }

    #[test]
    fn literal_prefix_of_pure_literal_pattern_is_the_whole_pattern() {
        assert_eq!(prefix("cat"), b"cat".to_vec());
        assert_eq!(prefix("c"), b"c".to_vec());
    }

    #[test]
    fn literal_prefix_stops_at_first_non_literal_atom() {
        assert_eq!(prefix("ca.*"), b"ca".to_vec());
        assert_eq!(prefix("ca*t"), b"c".to_vec());
        assert_eq!(prefix("ca?t"), b"c".to_vec());
    }

    #[test]
    fn literal_prefix_uses_a_mandatory_repeats_minimum_count() {
        // `a{2,3}` guarantees two leading `a`s; the old implementation gave
        // up at any quantifier and returned just `c`.
        assert_eq!(prefix("ca{2,3}t"), b"caa".to_vec());
        assert_eq!(prefix("ca+t"), b"ca".to_vec());
        assert_eq!(prefix("c(ab){2,}t"), b"cabab".to_vec());
        assert_eq!(prefix("ca{0,3}t"), b"c".to_vec());
    }

    #[test]
    fn literal_prefix_of_alternation_is_empty() {
        assert_eq!(prefix("cat|dog"), Vec::<u8>::new());
    }

    #[test]
    fn literal_prefix_of_leading_wildcard_atom_is_empty() {
        assert_eq!(prefix(".*cat"), Vec::<u8>::new());
        assert_eq!(prefix("[ab]cat"), Vec::<u8>::new());
        assert_eq!(prefix("@cat"), Vec::<u8>::new());
        assert_eq!(prefix("#"), Vec::<u8>::new());
        assert_eq!(prefix("<1-9>a"), Vec::<u8>::new());
    }

    #[test]
    fn literal_prefix_of_quoted_string_and_intersection() {
        assert_eq!(prefix(r#""cat".*"#), b"cat".to_vec());
        assert_eq!(prefix("cat.*&catalog"), b"catalog".to_vec());
    }

    #[test]
    fn literal_prefix_of_empty_pattern_is_empty() {
        assert_eq!(prefix(""), Vec::<u8>::new());
    }

    #[test]
    fn literal_prefix_is_a_true_prefix_of_every_match() {
        // Property check: whatever `literal_prefix` claims must actually be a
        // prefix of every term the pattern accepts, or `regexp_intersect`'s
        // binary-search narrowing would silently drop matches.
        let cases: &[(&str, &[&str])] = &[
            ("ca{2,3}t", &["caat", "caaat"]),
            ("ca+t", &["cat", "caaat"]),
            ("cat.*", &["cat", "catalog"]),
            (r#""cat"[0-9]"#, &["cat7"]),
            ("cat.*&catalog", &["catalog"]),
            ("c(ab){2,}t", &["cababt", "cabababt"]),
        ];
        for (pattern, terms) in cases {
            let p = RegexpPattern::parse(pattern).unwrap();
            let pfx = p.literal_prefix();
            for term in *terms {
                assert!(p.matches(term.as_bytes()), "{pattern} should match {term}");
                assert!(
                    term.as_bytes().starts_with(&pfx),
                    "{pattern}: prefix {pfx:?} is not a prefix of {term}"
                );
            }
        }
    }

    // -- dead-prefix skipping (the `ByteRunAutomaton` dead-state signal) ---

    /// The soundness property the whole skip rests on: if
    /// `could_match_prefix(p)` is false, then **no** string starting with `p`
    /// matches. Checked exhaustively over a small alphabet, which for these
    /// patterns covers every structurally distinct case.
    #[test]
    fn a_dead_prefix_has_no_matching_extension() {
        const ALPHABET: &[u8] = b"abz0";
        const MAX_LEN: usize = 4;
        let patterns = [
            "cat", "ca*t", "c.t", "a*z", "[ab]+z", "a{2,3}b", "(ab|az)+", "a.&.b", r"\d+z", "z@",
            "<1-30>b", "\"ab\"z", "a|bz", "#", "[^a]b",
        ];
        // Every string over ALPHABET up to MAX_LEN, shortest first.
        let mut words: Vec<Vec<u8>> = vec![Vec::new()];
        let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
        for _ in 0..MAX_LEN {
            let mut next = Vec::new();
            for w in &frontier {
                for &c in ALPHABET {
                    let mut e = w.clone();
                    e.push(c);
                    next.push(e);
                }
            }
            words.extend(next.iter().cloned());
            frontier = next;
        }

        for pattern in patterns {
            let p = RegexpPattern::parse(pattern).unwrap();
            for word in &words {
                if p.could_match_prefix(word) {
                    continue;
                }
                for candidate in &words {
                    assert!(
                        !(candidate.starts_with(word) && p.matches(candidate)),
                        "{pattern}: {:?} declared dead but {:?} matches",
                        String::from_utf8_lossy(word),
                        String::from_utf8_lossy(candidate)
                    );
                }
            }
        }
    }

    #[test]
    fn dead_prefix_len_finds_the_shortest_dead_prefix() {
        let p = RegexpPattern::parse("cat").unwrap();
        // "cat" itself is alive (it matches); "cats" dies at "cats" -- the
        // pattern is exhausted, so no longer term can match.
        assert_eq!(p.dead_prefix_len(b"cat"), None);
        assert_eq!(p.dead_prefix_len(b"cats"), Some(4));
        // A mismatch at the second byte kills the 2-byte prefix.
        assert_eq!(p.dead_prefix_len(b"cot"), Some(2));
        assert_eq!(p.dead_prefix_len(b"dog"), Some(1));

        // A trailing `.*` keeps every longer term alive.
        let p = RegexpPattern::parse("cat.*").unwrap();
        assert_eq!(p.dead_prefix_len(b"catalog"), None);
        assert_eq!(p.dead_prefix_len(b"cot"), Some(2));

        // `t1[0-9]` is the shape that makes this worth doing: exactly one
        // digit may follow, so a third byte kills the whole subtree.
        let p = RegexpPattern::parse("t1[0-9]").unwrap();
        assert_eq!(p.dead_prefix_len(b"t1"), None);
        assert_eq!(p.dead_prefix_len(b"t10"), None);
        assert_eq!(p.dead_prefix_len(b"t100"), Some(4));
        assert_eq!(p.dead_prefix_len(b"t1x"), Some(3));
    }

    #[test]
    fn dead_prefix_len_never_splits_a_codepoint() {
        // `é` is two bytes; a dead prefix that cut between them would skip
        // terms sharing only half a character.
        let p = RegexpPattern::parse("caf.").unwrap();
        let term = "caféx".as_bytes();
        let k = p
            .dead_prefix_len(term)
            .expect("term is longer than the pattern");
        assert!(
            std::str::from_utf8(&term[..k]).is_ok(),
            "dead prefix {k} splits a codepoint"
        );
    }

    #[test]
    fn an_all_accepting_pattern_has_no_dead_prefix() {
        let p = RegexpPattern::parse("@").unwrap();
        assert_eq!(p.dead_prefix_len(b"anything at all"), None);
        let p = RegexpPattern::parse(".*").unwrap();
        assert_eq!(p.dead_prefix_len(b"anything at all"), None);
    }

    #[test]
    fn the_empty_language_kills_every_prefix() {
        let p = RegexpPattern::parse("#").unwrap();
        assert!(!p.could_match_prefix(b""));
        assert_eq!(p.dead_prefix_len(b"a"), Some(0));
    }

    #[test]
    fn new_accepts_raw_bytes_and_decodes_them_leniently() {
        assert!(RegexpPattern::new(b"cat").unwrap().matches(b"cat"));
        // Ill-formed pattern bytes decode to U+FFFD rather than failing.
        assert!(RegexpPattern::new(&[0xFF]).is_ok());
    }
}
