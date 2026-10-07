//! A backtracking matcher with `java.util.regex.Pattern`'s semantics, for
//! the patterns [`super::java_regex`]'s `regex`-crate shim refuses:
//! backreferences, lookahead and lookbehind, atomic groups and possessive
//! quantifiers, `\b`/`\B`, `(?m)` anchors, `\G`, `\Z`, `\R`, comments mode
//! `(?x)`, `UNICODE_CHARACTER_CLASS` `(?U)`, `UNIX_LINES` `(?d)`, script,
//! block and binary properties, `\cX` and octal escapes, and the capture
//! semantics of repeated groups that can match empty.
//!
//! Re-specified from `Pattern`'s documented grammar and checked against it
//! on generated patterns and texts (`GenJavaRegex.java`); no JDK code is
//! copied. The pattern is parsed with Java's grammar into a tree, compiled
//! to a flat program and run over the text's UTF-16 units with explicit
//! stacks on the heap ([`vm`]), with the choices Java's compiler makes for
//! each construct:
//!
//! - `X{0,1}` is `X?` in every respect.
//! - A quantifier on a single character, class, assertion, backreference,
//!   lookaround or atomic group repeats that atom's *first* match at each
//!   step and backtracks only over the count (Java's `Curly`/`Ques`).
//! - A quantified group: `X?` is an alternation of the group and nothing;
//!   `X*`, `X+`, `X{n,m}` iterate a group whose body has a single way to
//!   match (no alternation or variable repetition: Java's `GroupCurly`) by
//!   its first match, recording each iteration's span, and any other group
//!   with full backtracking (`Loop`), which stops iterating -- even below
//!   the minimum -- after an iteration that matched empty; a possessive one
//!   repeats the whole group's first match.
//! - A greedy unbounded `Loop` that no quantified group encloses, in a
//!   pattern without backreferences, remembers for the whole `find()` the
//!   positions where one more iteration failed and goes straight to what
//!   follows when it is back at one (Java's guard against exponential
//!   backtracking). It changes no match, but it outlives a start position,
//!   so it changes which failed attempts leave captures behind.
//! - Lookbehind tries every start from `i - min` down to `i - max` (Java's
//!   window: lengths count a class as one unit, so a supplementary character
//!   needs `max` two), stepping by code points when the pattern holds a
//!   supplementary character after the lookbehind.
//! - `find()` tries every UTF-16 position, not before `len - min`; code
//!   point positions when the pattern's text holds a supplementary
//!   character or it has a single non-BMP literal or predicate (an escaped
//!   one inside a run of literals does not count).
//! - Captures set inside a lookahead stay set; a group's capture is undone
//!   when what follows it fails.
//!
//! Classes, literals and case folding reuse the shim's sets, which encode
//! Java's rules. A class or literal that holds only BMP characters reads one
//! UTF-16 unit, any other a code point.
//!
//! Speed, none of it observable: a run of nodes with one way to match
//! (characters, literals, backreferences, assertions, fixed repetitions) is
//! one instruction, and so is the minimum of a one-step repetition; a
//! capture of such a run checks the run after it before it records anything
//! to undo; an alternative -- or a search start -- whose first character
//! cannot be the text's is not tried, when trying it would fail before
//! setting a capture or a loop's memo (a table per ASCII unit says which).
//!
//! **Refused** (`IllegalArgument` starting [`super::java_regex::UNSUPPORTED`],
//! as before): `\X`, `\b{g}` (extended grapheme clusters, from the JDK's
//! Unicode version's segmentation rules), `\N{name}` (character names) and
//! `CANON_EQ` (`(?c)`). Java's `StackOverflowError` -- whose depth depends on
//! the thread's stack -- is an `IllegalState` error, from
//! [`super::java_regex::JavaMatcher::try_find`] or `compile`, never a Rust
//! stack overflow, which would abort the process (and the JVM over FFI): a
//! match past [`vm::MAX_ENTRIES`] backtracking entries (some hundreds of
//! thousands of iterations, where Java's 1 MiB stack holds some 1,500), a
//! parse deeper than [`DEEP_STACK_BUDGET`] allows. Neither is measured
//! against the caller's stack: a match's depth is heap, and a pattern that
//! nests more than [`SHALLOW_NESTING`] levels is parsed -- and, if its
//! lookarounds, atomic groups or quantified groups nest that deep, matched
//! -- on a thread with a stack of its own.

use std::collections::HashMap;

use super::java_regex::{
    class_literal_set, class_range_set, for_property, literal_set, CpSet, JFlags, UNSUPPORTED,
};
use super::java_regex_props as props;
use crate::java_character::{
    self as jc, code_point_at, get_type, is_letter_or_digit, to_lower_case, to_upper_case,
};
use crate::AnalysisError;

mod vm;

/// The stack a parse may use on the caller's thread: measured, not counted,
/// since a frame's size depends on the build. A pattern that might nest
/// deeper than [`SHALLOW_NESTING`] is parsed on a thread of its own
/// instead, so this never meets the caller's own stack limit.
const STACK_BUDGET: usize = 256 * 1024;

/// How deeply a pattern may nest (groups, classes, intersections; and the
/// sub-programs a match runs nested) to be parsed and matched on the
/// caller's stack: some tens of KiB, whatever the caller has left (a JVM
/// thread's whole stack is 1 MiB, `-Xss1m`).
pub(crate) const SHALLOW_NESTING: usize = 32;

/// A deeper pattern is parsed (and, if its sub-programs nest that deep,
/// matched) on a thread of this stack...
const DEEP_STACK: usize = 32 * 1024 * 1024;

/// ... and a parse within this budget, beyond which it fails as Java's
/// `StackOverflowError` would.
const DEEP_STACK_BUDGET: usize = 24 * 1024 * 1024;

/// `f` on a thread with [`DEEP_STACK`]; `None` when no thread can be had.
fn on_deep_stack<T: Send>(f: impl FnOnce() -> T + Send) -> Option<T> {
    std::thread::scope(|s| {
        std::thread::Builder::new()
            .name("java-regex-deep".into())
            .stack_size(DEEP_STACK)
            .spawn_scoped(s, f)
            .ok()
            .map(|h| h.join().unwrap_or_else(|p| std::panic::resume_unwind(p)))
    })
}

/// An upper bound on how deeply the parser recurses over `p` (after
/// `\Q..\E` removal): every unescaped `(`, `[` and `&` may open a level.
fn nesting_bound(p: &[u32]) -> usize {
    let mut n = 0usize;
    let mut i = 0;
    while i < p.len() {
        match char::from_u32(p[i]) {
            Some('\\') => i += 1,
            Some('(' | '[' | '&') => n += 1,
            _ => {}
        }
        i += 1;
    }
    n
}

/// The address of a local: how deep the stack is here.
#[inline(always)]
fn stack_addr() -> usize {
    let probe = 0u8;
    std::hint::black_box(std::ptr::addr_of!(probe)) as usize
}

const MAX_REPS: u32 = 0x7FFF_FFFF;

// ------------------------------------------------------------------ flags

const CASE_INSENSITIVE: u32 = 0x02;
const COMMENTS: u32 = 0x04;
const MULTILINE: u32 = 0x08;
const DOTALL: u32 = 0x20;
const UNICODE_CASE: u32 = 0x40;
const UNIX_LINES: u32 = 0x01;
const UNICODE_CHARACTER_CLASS: u32 = 0x100;

fn syntax(msg: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("PatternSyntaxException: {msg}"))
}

fn unsupported(what: &str, why: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("{UNSUPPORTED} `{what}`: {why}"))
}

// ------------------------------------------------------------------- tree

/// How a quantifier repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Greed {
    Greedy,
    Lazy,
    Possessive,
}

/// What a repetition runs (see the module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RepMode {
    /// `Curly` over an atom's first match.
    First,
    /// `Ques` (`?` on anything but a group): an atom's first match or
    /// nothing, an empty first match included.
    Ques,
    /// `GroupCurly`: a deterministic group, iteration spans recorded.
    GroupCurly,
    /// `Loop`/`LazyLoop`: full backtracking.
    Loop,
}

/// A single-character matcher: a set, read as one unit (`bmp`) or one code
/// point.
#[derive(Debug, Clone)]
struct CharSet {
    set: CpSet,
    bmp: bool,
    /// The set's ASCII members, one bit each (the common case, tested
    /// without the range search).
    ascii: u128,
}

impl CharSet {
    fn new(set: CpSet, bmp: bool) -> Self {
        let ascii = (0..128u32)
            .filter(|&c| set.contains(c))
            .fold(0u128, |m, c| m | 1 << c);
        CharSet { set, bmp, ascii }
    }

    #[inline]
    fn contains(&self, c: u32) -> bool {
        if c < 128 {
            self.ascii >> c & 1 == 1
        } else {
            self.set.contains(c)
        }
    }
}

#[derive(Debug, Clone)]
enum Node {
    Empty,
    Seq(Vec<Node>),
    /// Alternatives, and the characters each must start with when that is
    /// known (`annotate_alternatives`): an alternative a text's next `char`
    /// cannot start is not tried, which changes nothing but the time.
    Alt(Vec<Node>, Vec<Option<CharSet>>),
    /// A run of literals, one set per code point (`Slice`).
    Slice(Vec<CharSet>),
    Char(CharSet),
    /// A group: capturing (`Some(index)`) or not; its body.
    Group(Option<usize>, Box<Node>),
    LookAhead {
        negate: bool,
        body: Box<Node>,
    },
    LookBehind {
        negate: bool,
        body: Box<Node>,
        /// Java's `rmin`, `rmax` (`int`; see [`Info`]).
        min: i32,
        max: i32,
        by_code_point: bool,
    },
    /// `(?>X)`.
    Atomic(Box<Node>),
    Repeat {
        atom: Box<Node>,
        min: u32,
        max: u32,
        greed: Greed,
        mode: RepMode,
        /// For `GroupCurly`: the capture to record each iteration in; for
        /// a greedy unbounded `Loop`: its failed-position memo (`posIndex`),
        /// when it has one (see [`Parser::top_loops`]).
        capture: Option<usize>,
    },
    BackRef {
        group: usize,
        /// Case-insensitive: `Some(unicode)`.
        ci: Option<bool>,
    },
    /// `\A`, and `^` without `MULTILINE`.
    Begin,
    /// `\z`.
    End,
    /// `^` with `MULTILINE` (`unix`: `UNIX_LINES`).
    Caret {
        unix: bool,
    },
    /// `$` and `\Z`.
    Dollar {
        multiline: bool,
        unix: bool,
    },
    /// `\b` (`not`: `\B`).
    Bound {
        not: bool,
        unicode: bool,
    },
    /// `\G`.
    LastMatch,
    /// `\R`.
    LineEnding,
}

/// `TreeInfo`: the minimum and maximum lengths (in Java's units), and
/// whether the tree has one way to match. The maximum is Java's `int`: it
/// wraps where Java's sums wrap (a greedy `X*` on a character adds
/// `MAX_REPS`), and is invalid (`None`) where Java's `Curly` detects
/// overflow or a construct has no bound (backreferences, group loops).
#[derive(Debug, Clone, Copy)]
struct Info {
    min: usize,
    max: Option<i32>,
    deterministic: bool,
}

fn study(n: &Node) -> Info {
    let fixed = |l: usize| Info {
        min: l,
        max: Some(i32::try_from(l).unwrap_or(i32::MAX)),
        deterministic: true,
    };
    match n {
        Node::Empty
        | Node::LookAhead { .. }
        | Node::LookBehind { .. }
        | Node::Begin
        | Node::End
        | Node::Caret { .. }
        | Node::Dollar { .. }
        | Node::Bound { .. }
        | Node::LastMatch => fixed(0),
        Node::Slice(s) => fixed(s.len()),
        Node::Char(_) => fixed(1),
        Node::LineEnding => Info {
            min: 1,
            max: Some(2),
            deterministic: true,
        },
        Node::BackRef { .. } => Info {
            min: 0,
            max: None,
            deterministic: true,
        },
        Node::Seq(items) => items.iter().fold(fixed(0), |a, n| {
            let b = study(n);
            Info {
                min: a.min.saturating_add(b.min),
                max: seq_max(a.max, b.max, n),
                deterministic: a.deterministic && b.deterministic,
            }
        }),
        Node::Alt(alts, _) => {
            let infos: Vec<Info> = alts.iter().map(study).collect();
            Info {
                min: infos.iter().map(|i| i.min).min().unwrap_or(0),
                max: infos.iter().try_fold(0i32, |m, i| i.max.map(|x| m.max(x))),
                deterministic: alts.len() < 2 && infos.iter().all(|i| i.deterministic),
            }
        }
        Node::Group(_, b) | Node::Atomic(b) => study(b),
        Node::Repeat {
            atom,
            min,
            max,
            greed,
            mode,
            ..
        } => {
            let a = study(atom);
            if *mode == RepMode::Ques {
                return Info {
                    min: 0,
                    max: a.max,
                    deterministic: false,
                };
            }
            let minl = a.min.saturating_mul(*min as usize);
            // `CharPropertyGreedy` (a greedy `*` or `+` on one character)
            // adds `MAX_REPS` unchecked; `Curly`, `GroupCurly` check the
            // product and `Loop` has no bound.
            let char_greedy = *mode == RepMode::First
                && *greed == Greed::Greedy
                && *max == MAX_REPS
                && *min <= 1
                && matches!(**atom, Node::Char(_));
            let maxl = if char_greedy {
                Some(MAX_REPS as i32)
            } else if *mode == RepMode::Loop {
                None
            } else {
                a.max.and_then(|x| {
                    let p = i64::from(x) * i64::from(*max);
                    i32::try_from(p).ok()
                })
            };
            Info {
                min: minl,
                max: maxl,
                deterministic: *mode != RepMode::Loop && a.deterministic && min == max,
            }
        }
    }
}

/// The maximum of `prev` then `n` (whose own maximum is `own`): Java
/// adds unchecked, but a counted repetition (`Curly`, `GroupCurly`)
/// detects the sum's overflow.
fn seq_max(prev: Option<i32>, own: Option<i32>, n: &Node) -> Option<i32> {
    let (p, o) = (prev?, own?);
    match n {
        Node::Repeat {
            mode,
            greed,
            max,
            min,
            atom,
            ..
        } if *mode != RepMode::Ques
            && !(*mode == RepMode::First
                && *greed == Greed::Greedy
                && *max == MAX_REPS
                && *min <= 1
                && matches!(**atom, Node::Char(_))) =>
        {
            p.checked_add(o)
        }
        _ => Some(p.wrapping_add(o)),
    }
}

// ----------------------------------------------------------------- parser

/// Java's `ASCII.isSpace`.
fn ascii_space(c: u32) -> bool {
    matches!(c, 0x20 | 0x09..=0x0D)
}

fn is_hex(c: u32) -> bool {
    char::from_u32(c).is_some_and(|c| c.is_ascii_hexdigit())
}

fn is_ascii_digit(c: u32) -> bool {
    (u32::from(b'0')..=u32::from(b'9')).contains(&c)
}

fn is_ascii_alpha(c: u32) -> bool {
    char::from_u32(c).is_some_and(|c| c.is_ascii_alphabetic())
}

/// Java's horizontal whitespace `\h`.
fn horiz_ws() -> CpSet {
    CpSet::from_ranges(vec![
        (0x20, 0x20),
        (0x09, 0x09),
        (0xA0, 0xA0),
        (0x1680, 0x1680),
        (0x180E, 0x180E),
        (0x2000, 0x200A),
        (0x202F, 0x202F),
        (0x205F, 0x205F),
        (0x3000, 0x3000),
    ])
}

/// Java's vertical whitespace `\v`.
fn vert_ws() -> CpSet {
    CpSet::from_ranges(vec![(0x0A, 0x0D), (0x85, 0x85), (0x2028, 0x2029)])
}

fn ascii(spec: &[(u8, u8)]) -> CpSet {
    CpSet::from_ranges(
        spec.iter()
            .map(|&(a, b)| (u32::from(a), u32::from(b)))
            .collect(),
    )
}

/// A generated property set by key ([`props::PROPERTIES`]).
fn prop(key: &str) -> Option<CpSet> {
    props::PROPERTIES
        .iter()
        .find(|(k, _)| *k == key)
        .map(|(_, r)| CpSet::from_ranges(r.to_vec()))
}

/// The set a class escape (`\d \D \s \S \w \W \h \H \v \V`) stands for.
fn class_escape(c: u32, unicode: bool) -> Option<(CpSet, bool)> {
    let (set, neg) = match char::from_u32(c)? {
        'd' | 'D' if unicode => (prop("D")?, c == u32::from('D')),
        'd' | 'D' => (ascii(&[(b'0', b'9')]), c == u32::from('D')),
        's' | 'S' if unicode => (prop("S")?, c == u32::from('S')),
        's' | 'S' => (ascii(&[(b' ', b' '), (0x09, 0x0D)]), c == u32::from('S')),
        'w' | 'W' if unicode => (prop("W")?, c == u32::from('W')),
        'w' | 'W' => (
            ascii(&[(b'0', b'9'), (b'A', b'Z'), (b'_', b'_'), (b'a', b'z')]),
            c == u32::from('W'),
        ),
        'h' | 'H' => (horiz_ws(), c == u32::from('H')),
        'v' | 'V' => (vert_ws(), c == u32::from('V')),
        _ => return None,
    };
    // A negated class is never Java's BMP predicate; `\d \s \w \h \v`
    // without `UNICODE_CHARACTER_CLASS` are.
    let bmp = !neg && !unicode;
    Some(if neg {
        (set.complement(), false)
    } else {
        (set, bmp)
    })
}

/// What an escape read: a literal code point, or a node.
enum Escape {
    Literal(u32),
    Class(CharSet),
    Node(Node),
}

/// A class under construction: a set, and whether it is still a BMP
/// predicate.
#[derive(Clone)]
struct ClassAcc {
    set: CpSet,
    bmp: bool,
}

impl ClassAcc {
    fn union(self, o: ClassAcc) -> ClassAcc {
        ClassAcc {
            set: self.set.union(&o.set),
            bmp: self.bmp && o.bmp,
        }
    }

    fn and(self, o: ClassAcc) -> ClassAcc {
        ClassAcc {
            set: self.set.intersect(&o.set),
            bmp: self.bmp && o.bmp,
        }
    }
}

struct Parser<'p> {
    /// The pattern's code points, `\Q..\E` already turned into escapes.
    p: &'p [u32],
    cursor: usize,
    flags: u32,
    groups: usize,
    names: HashMap<String, usize>,
    /// The pattern holds a supplementary character (Java's
    /// `hasSupplementary`, which selects `StartS`).
    has_supplementary: bool,
    /// Greedy unbounded `Loop`s made so far.
    loops: usize,
    /// `topClosureNodes`: the greedy unbounded `Loop`s no quantified group
    /// encloses. Each remembers, for one `find`, the positions where
    /// another iteration failed and skips straight to what follows when it
    /// returns to one -- unless the pattern has a backreference
    /// (`hasGroupRef`). The memo outlives a start position, so it changes
    /// which failed attempts leave captures behind.
    top_loops: Vec<usize>,
    has_group_ref: bool,
    /// The stack's depth where the parse started, and how much it may use.
    stack_base: usize,
    budget: usize,
}

/// `RemoveQEQuoting`: `\Q..\E` becomes its characters, each non-alphanumeric
/// one escaped (an unclosed `\Q` quotes to the end).
fn remove_qe(p: &[u32]) -> Vec<u32> {
    let bs = u32::from(b'\\');
    let mut out = Vec::with_capacity(p.len());
    let mut i = 0;
    while i < p.len() {
        let c = p[i];
        if c == bs && p.get(i + 1) == Some(&u32::from(b'Q')) {
            i += 2;
            while i < p.len() {
                if p[i] == bs && p.get(i + 1) == Some(&u32::from(b'E')) {
                    i += 2;
                    break;
                }
                let alnum = char::from_u32(p[i]).is_some_and(|c| c.is_ascii_alphanumeric());
                if !alnum && p[i] < 0x80 {
                    out.push(bs);
                }
                if !alnum && p[i] >= 0x80 {
                    // A non-ASCII quoted character is itself: no escape needed.
                }
                out.push(p[i]);
                i += 1;
            }
            continue;
        }
        if c == bs && i + 1 < p.len() {
            out.push(c);
            out.push(p[i + 1]);
            i += 2;
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

impl Parser<'_> {
    /// Nesting that would take more than [`STACK_BUDGET`]: Java's
    /// `StackOverflowError` from `compile`.
    fn stack_check(&self) -> Result<(), AnalysisError> {
        if stack_addr().abs_diff(self.stack_base) > self.budget {
            return Err(overflow());
        }
        Ok(())
    }

    fn has(&self, f: u32) -> bool {
        self.flags & f != 0
    }

    fn jflags(&self) -> JFlags {
        JFlags {
            ci: self.has(CASE_INSENSITIVE),
            uc: self.has(UNICODE_CASE),
            dotall: self.has(DOTALL),
            multiline: self.has(MULTILINE),
        }
    }

    /// The code point at `i`, 0 past the end (Java's terminating zeros).
    fn at(&self, i: usize) -> u32 {
        self.p.get(i).copied().unwrap_or(0)
    }

    fn at_end(&self) -> bool {
        self.cursor >= self.p.len()
    }

    /// `peekPastWhitespace`: in comments mode, skips whitespace and `#`
    /// comments from the cursor.
    fn skip_comments(&mut self) {
        if !self.has(COMMENTS) {
            return;
        }
        loop {
            let c = self.at(self.cursor);
            if self.cursor < self.p.len() && ascii_space(c) {
                self.cursor += 1;
            } else if self.cursor < self.p.len() && c == u32::from(b'#') {
                while self.cursor < self.p.len()
                    && !matches!(self.at(self.cursor), 0x0A | 0x0D | 0x85 | 0x2028 | 0x2029)
                {
                    self.cursor += 1;
                }
            } else {
                return;
            }
        }
    }

    /// `peek()`.
    fn peek(&mut self) -> u32 {
        self.skip_comments();
        self.at(self.cursor)
    }

    /// `read()`: the next code point, consumed.
    fn read(&mut self) -> u32 {
        let c = self.peek();
        self.cursor += 1;
        c
    }

    /// `next()`: consumes one, peeks the one after.
    fn next(&mut self) -> u32 {
        self.cursor += 1;
        self.peek()
    }

    /// The code point after a backslash at the cursor, without comment
    /// skipping (`skip()`), consuming both.
    fn skip(&mut self) -> u32 {
        let c = self.at(self.cursor + 1);
        self.cursor += 2;
        c
    }

    fn accept(&mut self, c: u8, msg: &str) -> Result<(), AnalysisError> {
        if self.read() != u32::from(c) {
            return Err(syntax(msg));
        }
        Ok(())
    }

    /// `single(c)` and whether it is Java's BMP kind (`SingleU`, for a
    /// cased character under `(?iu)`, is not).
    fn single(&self, c: u32) -> (CpSet, bool) {
        let unicode_ci = self.has(CASE_INSENSITIVE)
            && self.has(UNICODE_CASE)
            && to_upper_case(c) != to_lower_case(to_upper_case(c));
        (
            literal_set(c, self.jflags(), false),
            c < 0x10000 && !unicode_ci,
        )
    }

    /// `expr()`: alternatives separated by `|`.
    fn expr(&mut self) -> Result<Node, AnalysisError> {
        let mut alts = Vec::new();
        loop {
            alts.push(self.sequence()?);
            if self.peek() != u32::from(b'|') {
                break;
            }
            self.cursor += 1;
        }
        Ok(if alts.len() == 1 {
            alts.pop().expect("one alternative")
        } else {
            Node::Alt(alts, Vec::new())
        })
    }

    /// `sequence()`.
    fn sequence(&mut self) -> Result<Node, AnalysisError> {
        let mut items = Vec::new();
        loop {
            let c = self.peek();
            let node = match char::from_u32(c).unwrap_or('\u{FFFD}') {
                '(' => match self.group()? {
                    None => continue,
                    Some(n) => {
                        items.push(n);
                        continue;
                    }
                },
                '[' => {
                    self.cursor += 1;
                    let acc = self.class(true)?;
                    self.char_node(acc.set, acc.bmp)
                }
                '\\' => {
                    let e = self.at(self.cursor + 1);
                    if e == u32::from(b'p') || e == u32::from(b'P') {
                        self.cursor += 2;
                        let one = if self.at(self.cursor) == u32::from(b'{') {
                            self.cursor += 1;
                            false
                        } else {
                            true
                        };
                        let acc = self.family(one, e == u32::from(b'P'))?;
                        self.char_node(acc.set, acc.bmp)
                    } else {
                        self.atom()?
                    }
                }
                '^' => {
                    self.cursor += 1;
                    if self.has(MULTILINE) {
                        Node::Caret {
                            unix: self.has(UNIX_LINES),
                        }
                    } else {
                        Node::Begin
                    }
                }
                '$' => {
                    self.cursor += 1;
                    Node::Dollar {
                        multiline: self.has(MULTILINE),
                        unix: self.has(UNIX_LINES),
                    }
                }
                '.' => {
                    self.cursor += 1;
                    let set = if self.has(DOTALL) {
                        CpSet::all()
                    } else if self.has(UNIX_LINES) {
                        CpSet::single(0x0A).complement()
                    } else {
                        CpSet::from_ranges(vec![
                            (0x0A, 0x0A),
                            (0x0D, 0x0D),
                            (0x85, 0x85),
                            (0x2028, 0x2029),
                        ])
                        .complement()
                    };
                    Node::Char(CharSet::new(set, false))
                }
                '|' | ')' => break,
                '?' | '*' | '+' => {
                    return Err(syntax(&format!(
                        "Dangling meta character '{}'",
                        char::from_u32(c).unwrap_or('?')
                    )))
                }
                _ if c == 0 && self.at_end() => break,
                _ => self.atom()?,
            };
            let node = self.closure(node)?;
            items.push(node);
        }
        Ok(match items.len() {
            0 => Node::Empty,
            1 => items.pop().expect("one item"),
            _ => Node::Seq(items),
        })
    }

    /// `atom()`: a run of literal characters (a slice, or one character),
    /// or the node of a single escape.
    fn atom(&mut self) -> Result<Node, AnalysisError> {
        let mut buf: Vec<u32> = Vec::new();
        let mut prev = self.cursor;
        let mut c = self.peek();
        loop {
            match char::from_u32(c).unwrap_or('\u{FFFD}') {
                '*' | '+' | '?' | '{' => {
                    if buf.len() > 1 {
                        self.cursor = prev;
                        buf.pop();
                    }
                    break;
                }
                '$' | '.' | '^' | '(' | '[' | '|' | ')' => break,
                '\\' => {
                    let e = self.at(self.cursor + 1);
                    if e == u32::from(b'p') || e == u32::from(b'P') {
                        // A slice is waiting (sequence() reads a leading \p).
                        break;
                    }
                    prev = self.cursor;
                    match self.escape(false, buf.is_empty(), false)? {
                        Escape::Literal(l) => {
                            // An escaped supplementary character (`\x{1F600}`)
                            // is no supplementary literal: only a single one,
                            // a non-BMP predicate, selects `StartS`.
                            buf.push(l);
                            c = self.peek();
                            continue;
                        }
                        Escape::Class(cs) if buf.is_empty() => {
                            return Ok(self.char_node(cs.set, cs.bmp))
                        }
                        Escape::Node(n) if buf.is_empty() => return Ok(n),
                        _ => {
                            // Unwind the escape: the slice ends before it.
                            self.cursor = prev;
                            break;
                        }
                    }
                }
                _ if c == 0 && self.at_end() => break,
                _ => {
                    prev = self.cursor;
                    buf.push(c);
                    c = self.next();
                    continue;
                }
            }
        }
        Ok(match buf.len() {
            1 => {
                let (set, bmp) = self.single(buf[0]);
                self.char_node(set, bmp)
            }
            _ => {
                let f = self.jflags();
                Node::Slice(
                    buf.iter()
                        .map(|&b| CharSet::new(literal_set(b, f, true), false))
                        .collect(),
                )
            }
        })
    }

    /// `newCharProperty`: a predicate that is not Java's BMP kind makes the
    /// pattern search code points (`hasSupplementary`).
    fn char_node(&mut self, set: CpSet, bmp: bool) -> Node {
        if !bmp {
            self.has_supplementary = true;
        }
        Node::Char(CharSet::new(set, bmp))
    }

    /// `escape(inclass, create, isrange)`: the escape at the cursor (the
    /// backslash).
    fn escape(
        &mut self,
        inclass: bool,
        create: bool,
        isrange: bool,
    ) -> Result<Escape, AnalysisError> {
        let c = self.skip();
        let unicode = self.has(UNICODE_CHARACTER_CLASS);
        let node = |n: Node| Ok(Escape::Node(n));
        let ch = char::from_u32(c).unwrap_or('\u{FFFD}');
        match ch {
            '0' => return Ok(Escape::Literal(self.octal()?)),
            '1'..='9' if !inclass => {
                if !create {
                    return node(Node::Empty);
                }
                return node(self.backref(c - u32::from(b'0')));
            }
            'A' if !inclass => return node(Node::Begin),
            'B' if !inclass => return node(Node::Bound { not: true, unicode }),
            'G' if !inclass => return node(Node::LastMatch),
            'R' if !inclass => return node(Node::LineEnding),
            'X' if !inclass => {
                return Err(unsupported(
                    "\\X",
                    "extended grapheme clusters follow the JDK's Unicode segmentation rules",
                ))
            }
            'Z' if !inclass => {
                return node(Node::Dollar {
                    multiline: false,
                    unix: self.has(UNIX_LINES),
                })
            }
            'z' if !inclass => return node(Node::End),
            'b' if !inclass => {
                if self.at(self.cursor) == u32::from(b'{')
                    && self.at(self.cursor + 1) == u32::from(b'g')
                {
                    if self.at(self.cursor + 2) != u32::from(b'}') {
                        return Err(syntax("Illegal/unsupported escape sequence"));
                    }
                    return Err(unsupported(
                        "\\b{g}",
                        "grapheme cluster boundaries follow the JDK's Unicode segmentation rules",
                    ));
                }
                return node(Node::Bound {
                    not: false,
                    unicode,
                });
            }
            'k' if !inclass => {
                if self.read() != u32::from(b'<') {
                    return Err(syntax(
                        "\\k is not followed by '<' for named capturing group",
                    ));
                }
                let first = self.read();
                let name = self.group_name(first)?;
                let Some(&g) = self.names.get(&name) else {
                    return Err(syntax(&format!(
                        "named capturing group <{name}> does not exist"
                    )));
                };
                return node(self.backref_node(g));
            }
            'N' => return Err(unsupported("\\N{..}", "character names are not ported")),
            'a' => return Ok(Escape::Literal(7)),
            'c' => {
                if self.cursor < self.p.len() {
                    let x = self.read();
                    return Ok(Escape::Literal(x ^ 64));
                }
                return Err(syntax("Illegal control escape sequence"));
            }
            'e' => return Ok(Escape::Literal(0x1B)),
            'f' => return Ok(Escape::Literal(0x0C)),
            'n' => return Ok(Escape::Literal(0x0A)),
            'r' => return Ok(Escape::Literal(0x0D)),
            't' => return Ok(Escape::Literal(0x09)),
            'u' => return Ok(Escape::Literal(self.unicode_escape()?)),
            'x' => return Ok(Escape::Literal(self.hex()?)),
            'v' if isrange => return Ok(Escape::Literal(0x0B)),
            'd' | 'D' | 's' | 'S' | 'w' | 'W' | 'h' | 'H' | 'v' | 'V' => {
                let (set, bmp) =
                    class_escape(c, unicode).ok_or_else(|| syntax("property tables"))?;
                return Ok(Escape::Class(CharSet::new(set, bmp)));
            }
            _ if ch.is_ascii_alphabetic() || ch.is_ascii_digit() => {}
            _ => return Ok(Escape::Literal(c)),
        }
        Err(syntax("Illegal/unsupported escape sequence"))
    }

    fn backref_node(&mut self, group: usize) -> Node {
        self.has_group_ref = true;
        Node::BackRef {
            group,
            ci: self.has(CASE_INSENSITIVE).then(|| self.has(UNICODE_CASE)),
        }
    }

    /// `ref(refNum)`: more digits while the group they name exists.
    fn backref(&mut self, first: u32) -> Node {
        let mut n = first as usize;
        loop {
            let c = self.peek();
            if !is_ascii_digit(c) {
                break;
            }
            let next = n * 10 + (c - u32::from(b'0')) as usize;
            if self.groups < next {
                break;
            }
            n = next;
            self.cursor += 1;
        }
        self.backref_node(n)
    }

    /// `o()`: `\0n`, `\0nn`, `\0mnn` (m <= 3).
    fn octal(&mut self) -> Result<u32, AnalysisError> {
        let oct = |c: u32| (u32::from(b'0')..=u32::from(b'7')).contains(&c);
        let n = self.read();
        if oct(n) {
            let m = self.read();
            if oct(m) {
                let o = self.read();
                if oct(o) && n <= u32::from(b'3') {
                    return Ok((n - 48) * 64 + (m - 48) * 8 + (o - 48));
                }
                self.cursor -= 1;
                return Ok((n - 48) * 8 + (m - 48));
            }
            self.cursor -= 1;
            return Ok(n - 48);
        }
        Err(syntax("Illegal octal escape sequence"))
    }

    /// `x()`: `\xhh` or `\x{h..h}`.
    fn hex(&mut self) -> Result<u32, AnalysisError> {
        let n = self.read();
        if is_hex(n) {
            let m = self.read();
            if is_hex(m) {
                return Ok(hex_val(n) * 16 + hex_val(m));
            }
        } else if n == u32::from(b'{') && is_hex(self.peek()) {
            let mut ch: u32 = 0;
            let mut d;
            loop {
                d = self.read();
                if !is_hex(d) {
                    break;
                }
                ch = (ch << 4) + hex_val(d);
                if ch > 0x10FFFF {
                    return Err(syntax("Hexadecimal codepoint is too big"));
                }
            }
            if d != u32::from(b'}') {
                return Err(syntax("Unclosed hexadecimal escape sequence"));
            }
            return Ok(ch);
        }
        Err(syntax("Illegal hexadecimal escape sequence"))
    }

    fn uxxxx(&mut self) -> Result<u32, AnalysisError> {
        let mut n = 0;
        for _ in 0..4 {
            let c = self.read();
            if !is_hex(c) {
                return Err(syntax("Illegal Unicode escape sequence"));
            }
            n = n * 16 + hex_val(c);
        }
        Ok(n)
    }

    /// `u()`: `\uhhhh`, two of them joined into a surrogate pair.
    fn unicode_escape(&mut self) -> Result<u32, AnalysisError> {
        let n = self.uxxxx()?;
        if (0xD800..=0xDBFF).contains(&n) {
            let save = self.cursor;
            if self.read() == u32::from(b'\\') && self.read() == u32::from(b'u') {
                if let Ok(n2) = self.uxxxx() {
                    if (0xDC00..=0xDFFF).contains(&n2) {
                        return Ok(0x10000 + ((n - 0xD800) << 10) + (n2 - 0xDC00));
                    }
                }
            }
            self.cursor = save;
        }
        Ok(n)
    }

    /// `groupname(ch)`: `[a-zA-Z][a-zA-Z0-9]*` then `>`.
    fn group_name(&mut self, first: u32) -> Result<String, AnalysisError> {
        if !is_ascii_alpha(first) {
            return Err(syntax(
                "capturing group name does not start with a Latin letter",
            ));
        }
        let mut s = String::new();
        let mut c = first;
        loop {
            s.push(char::from_u32(c).unwrap_or('?'));
            c = self.read();
            if !char::from_u32(c).is_some_and(|c| c.is_ascii_alphanumeric()) {
                break;
            }
        }
        if c != u32::from(b'>') {
            return Err(syntax("named capturing group is missing trailing '>'"));
        }
        Ok(s)
    }
}

fn hex_val(c: u32) -> u32 {
    char::from_u32(c).and_then(|c| c.to_digit(16)).unwrap_or(0)
}

impl Parser<'_> {
    /// `group0()`: `None` for a flags-only group `(?i)`.
    fn group(&mut self) -> Result<Option<Node>, AnalysisError> {
        self.stack_check()?;
        let save = self.flags;
        let saved_loops = self.top_loops.len();
        self.cursor += 1; // '('
                          // `next()`: with COMMENTS, `( ?:x)` is `(?:x)`.
        let node = if self.peek() == u32::from(b'?') {
            let c = self.skip_after_question();
            match char::from_u32(c).unwrap_or('\u{FFFD}') {
                ':' => Node::Group(None, Box::new(self.expr()?)),
                '=' | '!' => Node::LookAhead {
                    negate: c == u32::from(b'!'),
                    body: Box::new(self.expr()?),
                },
                '>' => Node::Atomic(Box::new(self.expr()?)),
                '<' => {
                    let c2 = self.read();
                    if c2 != u32::from(b'=') && c2 != u32::from(b'!') {
                        let name = self.group_name(c2)?;
                        if self.names.contains_key(&name) {
                            return Err(syntax(&format!(
                                "Named capturing group <{name}> is already defined"
                            )));
                        }
                        self.groups += 1;
                        let index = self.groups;
                        self.names.insert(name, index);
                        Node::Group(Some(index), Box::new(self.expr()?))
                    } else {
                        let start = self.cursor;
                        let body = self.expr()?;
                        let info = study(&body);
                        let Some(max) = info.max else {
                            return Err(syntax(
                                "Look-behind group does not have an obvious maximum length",
                            ));
                        };
                        let by_code_point = self.p[start.min(self.p.len())..]
                            .iter()
                            .any(|&c| c >= 0x10000);
                        Node::LookBehind {
                            negate: c2 == u32::from(b'!'),
                            body: Box::new(body),
                            min: i32::try_from(info.min).unwrap_or(0x0FFF_FFFF),
                            max,
                            by_code_point,
                        }
                    }
                }
                '$' | '@' => return Err(syntax("Unknown group type")),
                _ => {
                    self.cursor -= 1;
                    self.add_flags()?;
                    let c = self.read();
                    if c == u32::from(b')') {
                        // Inline modifier only: the flags stay for the rest
                        // of the enclosing group.
                        return Ok(None);
                    }
                    if c != u32::from(b':') {
                        return Err(syntax("Unknown inline modifier"));
                    }
                    Node::Group(None, Box::new(self.expr()?))
                }
            }
        } else {
            self.groups += 1;
            let index = self.groups;
            Node::Group(Some(index), Box::new(self.expr()?))
        };
        self.accept(b')', "Unclosed group")?;
        self.flags = save;
        let group = matches!(node, Node::Group(..));
        let quantified = matches!(char::from_u32(self.peek()), Some('?' | '*' | '+' | '{'));
        if group && quantified {
            // A quantified group's inner loops are no longer top-level
            // (a lookaround or atomic group keeps them); its own loop
            // may be.
            self.top_loops.truncate(saved_loops);
        }
        Ok(Some(self.closure(node)?))
    }

    /// After `(?`: the next code point, consumed (Java's `skip()`).
    fn skip_after_question(&mut self) -> u32 {
        let c = self.at(self.cursor + 1);
        self.cursor += 2;
        c
    }

    /// `addFlag()` / `subFlag()`.
    fn add_flags(&mut self) -> Result<(), AnalysisError> {
        let mut c = self.peek();
        let mut on = true;
        loop {
            let f = match char::from_u32(c).unwrap_or('\u{FFFD}') {
                'i' => CASE_INSENSITIVE,
                'm' => MULTILINE,
                's' => DOTALL,
                'd' => UNIX_LINES,
                'u' => UNICODE_CASE,
                'x' => COMMENTS,
                'U' => UNICODE_CHARACTER_CLASS | UNICODE_CASE,
                'c' => return Err(unsupported("(?c)", "CANON_EQ is not ported")),
                '-' if on => {
                    on = false;
                    c = self.next();
                    continue;
                }
                _ => return Ok(()),
            };
            if on {
                self.flags |= f;
            } else {
                self.flags &= !f;
            }
            c = self.next();
        }
    }

    /// `closure(prev)`: a quantifier, if one follows.
    fn closure(&mut self, prev: Node) -> Result<Node, AnalysisError> {
        let c = self.peek();
        let (min, max) = match char::from_u32(c).unwrap_or('\u{FFFD}') {
            '?' => (0, 1),
            '*' => (0, MAX_REPS),
            '+' => (1, MAX_REPS),
            '{' => {
                let d = self.at(self.cursor + 1);
                if !is_ascii_digit(d) {
                    return Err(syntax("Illegal repetition"));
                }
                self.cursor += 1;
                let mut cmin: i64 = 0;
                let mut ch = self.read();
                while is_ascii_digit(ch) {
                    cmin = cmin.saturating_mul(10).saturating_add(i64::from(ch - 48));
                    ch = self.read();
                }
                let mut cmax = cmin;
                if ch == u32::from(b',') {
                    ch = self.read();
                    cmax = i64::from(MAX_REPS);
                    if ch != u32::from(b'}') {
                        cmax = 0;
                        while is_ascii_digit(ch) {
                            cmax = cmax.saturating_mul(10).saturating_add(i64::from(ch - 48));
                            ch = self.read();
                        }
                    }
                }
                if ch != u32::from(b'}') {
                    return Err(syntax("Unclosed counted closure"));
                }
                if cmin > i64::from(MAX_REPS) || cmax > i64::from(MAX_REPS) || cmax < cmin {
                    return Err(syntax("Illegal repetition range"));
                }
                self.cursor -= 1;
                (cmin as u32, cmax as u32)
            }
            _ => return Ok(prev),
        };
        // `X{0,1}` is `X?` in every respect (black-box: captures, greed,
        // zero-width bodies).
        let ques = (min, max) == (0, 1);
        let n = self.next();
        let greed = if n == u32::from(b'?') {
            self.cursor += 1;
            Greed::Lazy
        } else if n == u32::from(b'+') {
            self.cursor += 1;
            Greed::Possessive
        } else {
            Greed::Greedy
        };
        let mut node = repeat(prev, min, max, greed, ques);
        if let Node::Repeat {
            mode: RepMode::Loop,
            greed: Greed::Greedy,
            max: MAX_REPS,
            capture,
            ..
        } = &mut node
        {
            *capture = Some(self.loops);
            self.top_loops.push(self.loops);
            self.loops += 1;
        }
        Ok(node)
    }
}

/// The node a quantifier makes of `atom` (see the module docs).
fn repeat(atom: Node, min: u32, max: u32, greed: Greed, ques: bool) -> Node {
    let first = |atom| Node::Repeat {
        atom: Box::new(atom),
        min,
        max,
        greed,
        mode: if ques { RepMode::Ques } else { RepMode::First },
        capture: None,
    };
    let Node::Group(index, body) = atom else {
        return first(atom);
    };
    if greed == Greed::Possessive {
        return first(Node::Group(index, body));
    }
    if ques {
        // `X?` of a group is an alternation with nothing.
        let group = Node::Group(index, body);
        return Node::Alt(
            match greed {
                Greed::Greedy => vec![group, Node::Empty],
                _ => vec![Node::Empty, group],
            },
            Vec::new(),
        );
    }
    if study(&body).deterministic {
        Node::Repeat {
            atom: body,
            min,
            max,
            greed,
            mode: RepMode::GroupCurly,
            capture: index,
        }
    } else {
        Node::Repeat {
            atom: Box::new(Node::Group(index, body)),
            min,
            max,
            greed,
            mode: RepMode::Loop,
            capture: None,
        }
    }
}

impl Parser<'_> {
    /// `clazz(consume)`, the cursor after the `[`.
    fn class(&mut self, consume: bool) -> Result<ClassAcc, AnalysisError> {
        self.stack_check()?;
        let mut prev: Option<ClassAcc> = None;
        let mut bits: Option<ClassAcc> = None;
        let mut neg = false;
        let mut c = self.peek();
        if c == u32::from(b'^') && self.at(self.cursor.wrapping_sub(1)) == u32::from(b'[') {
            c = self.next();
            neg = true;
        }
        loop {
            let ch = char::from_u32(c).unwrap_or('\u{FFFD}');
            if ch == '[' {
                self.cursor += 1;
                let curr = self.class(true)?;
                prev = Some(match prev {
                    None => curr,
                    Some(p) => p.union(curr),
                });
                c = self.peek();
                continue;
            }
            if ch == '&' && self.at(self.cursor + 1) == u32::from(b'&') {
                self.cursor += 2;
                c = self.peek();
                let mut right: Option<ClassAcc> = None;
                while c != u32::from(b']') && c != u32::from(b'&') {
                    if c == 0 && self.at_end() {
                        return Err(syntax("Unclosed character class"));
                    }
                    let r = if c == u32::from(b'[') {
                        self.cursor += 1;
                        self.class(true)?
                    } else {
                        self.class(false)?
                    };
                    right = Some(match right {
                        None => r,
                        Some(x) => x.union(r),
                    });
                    c = self.peek();
                }
                if let Some(b) = bits.take() {
                    prev = Some(match prev {
                        None => b,
                        Some(p) => p.union(b),
                    });
                }
                prev = match (prev, right) {
                    (None, None) => return Err(syntax("Bad class syntax")),
                    (None, Some(r)) => Some(r),
                    (Some(p), Some(r)) => Some(p.and(r)),
                    // `[a&&]`: Java intersects the left side with itself.
                    (Some(p), None) => Some(p),
                };
                continue;
            }
            if c == 0 && self.at_end() {
                return Err(syntax("Unclosed character class"));
            }
            if ch == ']' && (prev.is_some() || bits.is_some()) {
                if consume {
                    self.cursor += 1;
                }
                let mut r = match (prev, bits) {
                    (Some(p), Some(b)) => p.union(b),
                    (Some(p), None) => p,
                    (None, Some(b)) => b,
                    (None, None) => unreachable!("checked above"),
                };
                if neg {
                    r = ClassAcc {
                        set: r.set.complement(),
                        bmp: false,
                    };
                }
                return Ok(r);
            }
            let (curr, is_bits) = self.range()?;
            if is_bits {
                bits = Some(match bits {
                    None => curr,
                    Some(b) => b.union(curr),
                });
            } else {
                prev = Some(match prev {
                    None => curr,
                    Some(p) => p.union(curr),
                });
            }
            c = self.peek();
        }
    }

    /// `range(bits)`: one class item; `true` when it went to the `BitClass`
    /// (a literal below U+0100).
    fn range(&mut self) -> Result<(ClassAcc, bool), AnalysisError> {
        let f = self.jflags();
        let mut c = self.peek();
        if c == u32::from(b'\\') {
            let e = self.at(self.cursor + 1);
            if e == u32::from(b'p') || e == u32::from(b'P') {
                self.cursor += 2;
                let one = if self.at(self.cursor) == u32::from(b'{') {
                    self.cursor += 1;
                    false
                } else {
                    true
                };
                return Ok((self.family(one, e == u32::from(b'P'))?, false));
            }
            let isrange = self.at(self.cursor + 2) == u32::from(b'-');
            match self.escape(true, true, isrange)? {
                Escape::Literal(l) => c = l,
                Escape::Class(cs) => {
                    return Ok((
                        ClassAcc {
                            set: cs.set,
                            bmp: cs.bmp,
                        },
                        false,
                    ))
                }
                Escape::Node(_) => return Err(syntax("Illegal/unsupported escape sequence")),
            }
        } else {
            self.cursor += 1;
        }
        if self.peek() == u32::from(b'-') {
            let end = self.at(self.cursor + 1);
            if end != u32::from(b'[') && end != u32::from(b']') {
                self.cursor += 1;
                let mut m = self.peek();
                if m == u32::from(b'\\') {
                    match self.escape(true, false, true)? {
                        Escape::Literal(l) => m = l,
                        _ => return Err(syntax("Illegal character range")),
                    }
                } else {
                    self.cursor += 1;
                }
                if m < c {
                    return Err(syntax("Illegal character range"));
                }
                // `CIRange`/`CIRangeU` are not Java's BMP kind.
                return Ok((
                    ClassAcc {
                        set: class_range_set(c, m, f),
                        bmp: m < 0x10000 && !f.ci,
                    },
                    false,
                ));
            }
        }
        let set = class_literal_set(c, f);
        let bits = c < 0x100
            && !(f.ci
                && f.uc
                && [0xFF, 0xB5, 0x49, 0x69, 0x53, 0x73, 0x4B, 0x6B, 0xC5, 0xE5].contains(&c));
        // The `BitClass` is a BMP predicate; otherwise `single(c)`.
        let bmp = bits || self.single(c).1;
        Ok((ClassAcc { set, bmp }, bits))
    }

    /// `family(singleLetter, isComplement)`, the cursor after `\p` (and
    /// `{` unless `one`).
    fn family(&mut self, one: bool, complement: bool) -> Result<ClassAcc, AnalysisError> {
        let name: String = if one {
            let c = self.at(self.cursor);
            self.cursor += 1;
            char::from_u32(c).map(String::from).unwrap_or_default()
        } else {
            let start = self.cursor;
            while self.cursor < self.p.len() && self.p[self.cursor] != u32::from(b'}') {
                self.cursor += 1;
            }
            if self.cursor >= self.p.len() {
                return Err(syntax("Unclosed character family"));
            }
            if start == self.cursor {
                return Err(syntax("Empty character family"));
            }
            let n: String = self.p[start..self.cursor]
                .iter()
                .filter_map(|&c| char::from_u32(c))
                .collect();
            self.cursor += 1;
            n
        };
        let ci = self.has(CASE_INSENSITIVE);
        let set = if let Some((key, value)) = name.split_once('=') {
            let set = match key.to_ascii_lowercase().as_str() {
                "sc" | "script" => script(value),
                "blk" | "block" => block(value),
                "gc" | "general_category" => for_property(value, ci),
                _ => None,
            };
            set.ok_or_else(|| {
                syntax(&format!(
                    "Unknown Unicode property {{name=<{key}>, value=<{value}>}}"
                ))
            })?
        } else {
            let set = if let Some(b) = name.strip_prefix("In") {
                block(b)
            } else if let Some(short) = name.strip_prefix("Is") {
                unicode_property(short, ci)
                    .or_else(|| java_property(short, ci))
                    .or_else(|| for_property(short, ci))
                    .or_else(|| script(short))
            } else {
                let posix = if self.has(UNICODE_CHARACTER_CLASS) {
                    posix_property(&name, ci)
                } else {
                    None
                };
                posix
                    .or_else(|| java_property(&name, ci))
                    .or_else(|| for_property(&name, ci))
            };
            set.ok_or_else(|| syntax(&format!("Unknown character property name {{{name}}}")))?
        };
        // The ASCII POSIX classes are Java's BMP predicates; categories,
        // scripts, blocks and properties are not.
        let posix_ascii = !self.has(UNICODE_CHARACTER_CLASS)
            && matches!(
                name.as_str(),
                "ASCII"
                    | "Alnum"
                    | "Alpha"
                    | "Blank"
                    | "Cntrl"
                    | "Digit"
                    | "Graph"
                    | "Lower"
                    | "Print"
                    | "Punct"
                    | "Space"
                    | "Upper"
                    | "XDigit"
            );
        Ok(if complement {
            ClassAcc {
                set: set.complement(),
                bmp: false,
            }
        } else {
            ClassAcc {
                set,
                bmp: posix_ascii,
            }
        })
    }
}

/// `forUnicodeScript(name)`.
fn script(name: &str) -> Option<CpSet> {
    let upper = name.to_ascii_uppercase();
    let id = props::SCRIPT_NAMES.iter().find(|(n, _)| *n == upper)?.1;
    let runs = &props::SCRIPT_RUNS;
    let ranges = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.1 == id)
        .map(|(i, r)| (r.0, runs.get(i + 1).map_or(0x10FFFF, |n| n.0 - 1)))
        .collect();
    Some(CpSet::from_ranges(ranges))
}

/// `forUnicodeBlock(name)`.
fn block(name: &str) -> Option<CpSet> {
    let upper = name.to_ascii_uppercase();
    let id = props::BLOCK_NAMES.iter().find(|(n, _)| *n == upper)?.1;
    let runs = &props::BLOCK_RUNS;
    let ranges = runs
        .iter()
        .enumerate()
        .filter(|(_, r)| r.1 == id)
        .map(|(i, r)| (r.0, runs.get(i + 1).map_or(0x10FFFF, |n| n.0 - 1)))
        .collect();
    Some(CpSet::from_ranges(ranges))
}

/// `forUnicodeProperty(name, caseIns)`: the binary properties.
fn unicode_property(name: &str, ci: bool) -> Option<CpSet> {
    let upper = name.to_ascii_uppercase();
    let key = match upper.as_str() {
        "HEXDIGIT" => "HEX_DIGIT",
        "JOINCONTROL" => "JOIN_CONTROL",
        "NONCHARACTERCODEPOINT" => "NONCHARACTER_CODE_POINT",
        "WHITESPACE" => "WHITE_SPACE",
        other => other,
    };
    if ci {
        if let Some(s) = prop(&format!("Ui:{key}")) {
            return Some(s);
        }
    }
    prop(&format!("U:{key}"))
}

/// The `java*` properties.
fn java_property(name: &str, ci: bool) -> Option<CpSet> {
    if !name.starts_with("java") {
        return None;
    }
    if ci {
        if let Some(s) = prop(&format!("Ji:{name}")) {
            return Some(s);
        }
    }
    prop(&format!("J:{name}"))
}

/// `forPOSIXName(name, caseIns)` (under `UNICODE_CHARACTER_CLASS`).
fn posix_property(name: &str, ci: bool) -> Option<CpSet> {
    let upper = name.to_ascii_uppercase();
    if ci {
        if let Some(s) = prop(&format!("Pi:{upper}")) {
            return Some(s);
        }
    }
    prop(&format!("P:{upper}"))
}

// ---------------------------------------------------------------- program

/// A compiled pattern for the backtracking matcher.
#[derive(Debug)]
pub(crate) struct Program {
    code: vm::Code,
    /// Capturing groups, group 0 included.
    groups: usize,
    names: HashMap<String, usize>,
    /// `StartS`: search positions skip the low half of a pair.
    has_supplementary: bool,
    min_len: usize,
    /// The characters every match starts with, when an attempt at any other
    /// fails at no cost: such a start is not tried.
    root_first: Option<CharSet>,
}

/// `Pattern.compile(pattern)` for the backtracking matcher.
pub(crate) fn compile(pattern: &str) -> Result<Program, AnalysisError> {
    let cps: Vec<u32> = pattern.chars().map(u32::from).collect();
    let p = remove_qe(&cps);
    if nesting_bound(&p) <= SHALLOW_NESTING {
        return compile_within(&p, STACK_BUDGET);
    }
    on_deep_stack(|| compile_within(&p, DEEP_STACK_BUDGET)).unwrap_or_else(|| Err(overflow()))
}

fn compile_within(p: &[u32], budget: usize) -> Result<Program, AnalysisError> {
    let mut parser = Parser {
        p,
        cursor: 0,
        flags: 0,
        groups: 0,
        names: HashMap::new(),
        has_supplementary: false,
        loops: 0,
        top_loops: Vec::new(),
        has_group_ref: false,
        stack_base: stack_addr(),
        budget,
    };
    // A supplementary character in the pattern's text, escaped or quoted
    // or not.
    parser.has_supplementary = p.iter().any(|&c| c >= 0x10000);
    let mut root = parser.expr()?;
    if parser.peek() != 0 || !parser.at_end() {
        if parser.peek() == u32::from(b')') {
            return Err(syntax("Unmatched closing ')'"));
        }
        return Err(syntax("Unexpected internal error"));
    }
    if p.iter()
        .rev()
        .take_while(|&&c| c == u32::from(b'\\'))
        .count()
        % 2
        == 1
    {
        return Err(syntax("Unescaped trailing backslash"));
    }
    let min_len = study(&root).min;
    let memo: Vec<bool> = (0..parser.loops)
        .map(|id| !parser.has_group_ref && parser.top_loops.contains(&id))
        .collect();
    keep_memos(&mut root, &memo);
    annotate_alternatives(&mut root);
    let root_first = first_chars(&root);
    Ok(Program {
        groups: parser.groups + 1,
        names: parser.names,
        has_supplementary: parser.has_supplementary,
        min_len,
        code: vm::Code::compile(&root),
        root_first,
    })
}

/// The characters every match of `n` starts with, when `n` must consume
/// one first and fails at no cost (no capture or memo set) otherwise.
fn first_chars(n: &Node) -> Option<CharSet> {
    match n {
        Node::Char(cs) => Some(cs.clone()),
        Node::Slice(sets) => sets.first().cloned(),
        Node::Group(_, body) => first_chars(body),
        // Zero-width items without side effects first: the first consuming
        // item's.
        Node::Seq(items) => first_chars(items.iter().find(|n| !inert(n))?),
        Node::Repeat { atom, min, .. } if *min >= 1 => first_chars(atom),
        Node::Alt(alts, _) => {
            let mut set = CpSet::default();
            for a in alts {
                set = set.union(&first_chars(a)?.set);
            }
            Some(CharSet::new(set, false))
        }
        _ => None,
    }
}

/// A zero-width node that sets nothing: an assertion, or a lookaround with
/// no capturing group and no memoised loop inside.
fn inert(n: &Node) -> bool {
    match n {
        Node::Begin
        | Node::End
        | Node::Caret { .. }
        | Node::Dollar { .. }
        | Node::Bound { .. }
        | Node::LastMatch => true,
        Node::LookAhead { body, .. } | Node::LookBehind { body, .. } => !sets_state(body),
        _ => false,
    }
}

/// Whether matching `n` can set a capture or a loop's memo.
fn sets_state(n: &Node) -> bool {
    match n {
        Node::Group(Some(_), _) => true,
        Node::Repeat {
            mode: RepMode::Loop,
            capture: Some(_),
            ..
        }
        | Node::Repeat {
            mode: RepMode::GroupCurly,
            capture: Some(_),
            ..
        } => true,
        Node::Seq(v) | Node::Alt(v, _) => v.iter().any(sets_state),
        Node::Group(None, b)
        | Node::Atomic(b)
        | Node::LookAhead { body: b, .. }
        | Node::LookBehind { body: b, .. } => sets_state(b),
        Node::Repeat { atom, .. } => sets_state(atom),
        _ => false,
    }
}

/// Fills each alternation's first characters, innermost first.
fn annotate_alternatives(n: &mut Node) {
    match n {
        Node::Seq(v) => v.iter_mut().for_each(annotate_alternatives),
        Node::Alt(v, firsts) => {
            v.iter_mut().for_each(annotate_alternatives);
            *firsts = v.iter().map(first_chars).collect();
        }
        Node::Group(_, b)
        | Node::Atomic(b)
        | Node::LookAhead { body: b, .. }
        | Node::LookBehind { body: b, .. } => annotate_alternatives(b),
        Node::Repeat { atom, .. } => annotate_alternatives(atom),
        _ => {}
    }
}

/// Drops the memo of each `Loop` that is not top-level after all.
fn keep_memos(n: &mut Node, keep: &[bool]) {
    match n {
        Node::Seq(v) | Node::Alt(v, _) => v.iter_mut().for_each(|c| keep_memos(c, keep)),
        Node::Group(_, b)
        | Node::Atomic(b)
        | Node::LookAhead { body: b, .. }
        | Node::LookBehind { body: b, .. } => keep_memos(b, keep),
        Node::Repeat {
            atom,
            mode,
            capture,
            ..
        } => {
            if *mode == RepMode::Loop && capture.is_some_and(|id| !keep[id]) {
                *capture = None;
            }
            keep_memos(atom, keep)
        }
        _ => {}
    }
}

impl Program {
    /// Capturing groups, group 0 included.
    pub(crate) fn group_total(&self) -> usize {
        self.groups
    }

    /// A named group's number.
    pub(crate) fn group_named(&self, name: &str) -> Option<usize> {
        self.names.get(name).copied()
    }
}

// ---------------------------------------------------------------- matcher

/// A match's state: the text, the groups (`2n`, `2n + 1`; -1 unset), where
/// the last match ended (`\G`), and the machine's stacks.
pub(crate) struct State<'t> {
    text: &'t [u16],
    pub(crate) groups: Vec<i32>,
    old_last: usize,
    scratch: vm::Scratch,
}

/// A hasher for the loops' failed-position memo: FxHash's mix over the
/// (loop, position) pairs, on the hot path of every memoised iteration.
pub(crate) type FastHash = crate::char_array_set::WordHash;

pub(crate) use vm::Scratch;

fn line_terminator(c: u16) -> bool {
    matches!(c, 0x0A | 0x0D | 0x85 | 0x2028 | 0x2029)
}

impl<'t> State<'t> {
    pub(crate) fn new(text: &'t [u16], groups: usize, old_last: usize) -> Self {
        Self::reusing(text, Vec::new(), Scratch::default(), groups, old_last)
    }

    /// A fresh state whose group slots and stacks reuse `buffer`'s and
    /// `scratch`'s allocations ([`Self::into_parts`] gives them back).
    pub(crate) fn reusing(
        text: &'t [u16],
        mut buffer: Vec<i32>,
        scratch: Scratch,
        groups: usize,
        old_last: usize,
    ) -> Self {
        buffer.clear();
        buffer.resize(groups * 2, -1);
        State {
            text,
            groups: buffer,
            old_last,
            scratch,
        }
    }

    /// The group slots and the stacks, for the next match to reuse.
    pub(crate) fn into_parts(self) -> (Vec<i32>, Scratch) {
        (self.groups, self.scratch)
    }

    fn len(&self) -> usize {
        self.text.len()
    }

    fn cp(&self, i: usize) -> u32 {
        code_point_at(self.text, i, self.text.len())
    }

    /// `Character.codePointBefore(seq, i)`.
    fn cp_before(&self, i: usize) -> u32 {
        let lo = self.text[i - 1];
        if (0xDC00..=0xDFFF).contains(&lo)
            && i >= 2
            && (0xD800..=0xDBFF).contains(&self.text[i - 2])
        {
            return code_point_at(self.text, i - 2, self.text.len());
        }
        u32::from(lo)
    }

    fn set_group(&mut self, g: usize, s: i32, e: i32) {
        if let Some(slot) = self.groups.get_mut(2 * g..2 * g + 2) {
            slot[0] = s;
            slot[1] = e;
        }
    }

    fn group(&self, g: usize) -> (i32, i32) {
        match self.groups.get(2 * g..2 * g + 2) {
            Some(s) => (s[0], s[1]),
            None => (-1, -1),
        }
    }
}

fn overflow() -> AnalysisError {
    AnalysisError::IllegalState(
        "StackOverflowError: java.util.regex needs more stack than it may use".to_string(),
    )
}

/// `countChars(seq, index, lengthInCodePoints)`: the UTF-16 units of that
/// many code points after `index` (or, negative, before it, not past 0).
fn count_chars(st: &State, index: i32, n: i32) -> i32 {
    let len = st.len() as i32;
    let x0 = index.clamp(0, len);
    if n == 1 && x0 < len && !(0xD800..=0xDBFF).contains(&st.text[x0 as usize]) {
        return 1;
    }
    let mut x = x0;
    if n >= 0 {
        let mut c = 0;
        while x < len && c < n {
            let hi = (0xD800..=0xDBFF).contains(&st.text[x as usize]);
            x += 1;
            if hi && x < len && (0xDC00..=0xDFFF).contains(&st.text[x as usize]) {
                x += 1;
            }
            c += 1;
        }
        return x - x0;
    }
    if x == 0 {
        return 0;
    }
    let want = n.wrapping_neg();
    let mut c = 0;
    while x > 0 && c < want {
        x -= 1;
        if (0xDC00..=0xDFFF).contains(&st.text[x as usize])
            && x > 0
            && (0xD800..=0xDBFF).contains(&st.text[x as usize - 1])
        {
            x -= 1;
        }
        c += 1;
    }
    x0 - x
}

/// `Bound.isWord`: `\w`'s characters.
fn is_word(c: u32, unicode: bool) -> bool {
    if unicode {
        return WORD_U.contains(c);
    }
    matches!(c, 0x30..=0x39 | 0x41..=0x5A | 0x5F | 0x61..=0x7A)
}

static WORD_U: std::sync::LazyLock<CpSet> =
    std::sync::LazyLock::new(|| prop("W").unwrap_or_default());

/// `Bound.hasBaseCharacter`: walks back over non-spacing marks from `i` to
/// a letter or digit.
fn has_base(st: &State, i: usize) -> bool {
    let mut x = i;
    loop {
        let c = st.cp(x);
        if is_letter_or_digit(c) {
            return true;
        }
        if get_type(c) != jc::NON_SPACING_MARK || x == 0 {
            return false;
        }
        x -= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// UTF-16 units of `s`, reading `\n`, `\r`, `\t`, `\\` and `\uXXXX`
    /// (a lone surrogate too).
    fn units(s: &str) -> Vec<u16> {
        let mut out = Vec::new();
        let mut it = s.chars().peekable();
        while let Some(c) = it.next() {
            if c != '\\' {
                let mut b = [0; 2];
                out.extend_from_slice(c.encode_utf16(&mut b));
                continue;
            }
            match it.next() {
                Some('n') => out.push(0x0A),
                Some('r') => out.push(0x0D),
                Some('t') => out.push(0x09),
                Some('u') => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    out.push(u16::from_str_radix(&hex, 16).unwrap());
                }
                Some(other) => out.extend([0x5C, other as u16]),
                None => out.push(0x5C),
            }
        }
        out
    }

    /// Every `find()` of `p` over `text`, as `Matcher` reports them, or the
    /// error.
    fn finds(p: &str, text: &[u16]) -> String {
        let prog = match compile(p) {
            Ok(prog) => prog,
            Err(e) if super::super::java_regex::is_unsupported(&e) => return "UNSUPPORTED".into(),
            Err(e) => return format!("ERR {e}"),
        };
        let mut out = String::new();
        let (mut from, mut old_last) = (0, 0);
        while from <= text.len() {
            let mut st = State::new(text, prog.group_total(), old_last);
            if !prog.search(from, &mut st).unwrap() {
                break;
            }
            let g = &st.groups;
            out.push_str(&format!("({},{}", g[0], g[1]));
            for c in g[2..].chunks(2) {
                out.push_str(&format!(" {}:{}", c[0], c[1]));
            }
            out.push(')');
            let (s, e) = (g[0] as usize, g[1] as usize);
            from = if s == e { e + 1 } else { e };
            old_last = e;
        }
        out
    }

    /// Escapes, classes, properties, flags, group syntax and the rarer
    /// matcher paths, against `Pattern` (JDK 25; the answers do not differ
    /// under JDK 21). `ERR x`: Java's `PatternSyntaxException` description.
    #[test]
    fn constructs_match_java() {
        let cases: &[(&str, &str, &str)] = &[
            (r"(?U)\w+\W", r"aé1_ -", r"(0,5)"),
            (r"\h+\H\v\V", r"a \t\nb", r""),
            (r"\pL+\PL", r"ab1", r"(0,3)"),
            (r"\Aa", r"aa", r"(0,1)"),
            (r"\b{gx}", r"a", r"ERR Illegal/unsupported escape sequence"),
            (r"\b{g}", r"a", r"UNSUPPORTED"),
            (
                r"(a)\ka",
                r"aa",
                r"ERR \k is not followed by '<' for named capturing group",
            ),
            (
                r"(a)\k<b>",
                r"aa",
                r"ERR named capturing group <b> does not exist",
            ),
            (r"\a\e\f", r"\u0007\u001B\u000C", r"(0,3)"),
            (r"\c", r"a", r"ERR Illegal control escape sequence"),
            (r"\cJ", r"\n", r"(0,1)"),
            (
                r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)\11",
                r"abcdefghijkk",
                r"(0,12 0:1 1:2 2:3 3:4 4:5 5:6 6:7 7:8 8:9 9:10 10:11)",
            ),
            (
                r"(a)(b)(c)(d)(e)(f)(g)(h)(i)(j)(k)\111",
                r"abcdefghijkk1",
                r"(0,13 0:1 1:2 2:3 3:4 4:5 5:6 6:7 7:8 8:9 9:10 10:11)",
            ),
            (r"\07\018\0400", r"\u0007\u00018 0", r"(0,5)"),
            (r"\x41\x{1F600}\x{41}", r"A😀A", r"(0,4)"),
            (r"\x{110000}", r"a", r"ERR Hexadecimal codepoint is too big"),
            (r"\x{41", r"a", r"ERR Unclosed hexadecimal escape sequence"),
            (r"\xG", r"a", r"ERR Illegal hexadecimal escape sequence"),
            (r"\uD83D\uDE00\uD83Dx", r"😀\uD83Dx", r"(0,4)"),
            (r"\uD83D\u0041", r"\uD83DA", r"(0,2)"),
            (
                r"(?<1a>x)",
                r"x",
                r"ERR capturing group name does not start with a Latin letter",
            ),
            (
                r"(?<a-b>x)",
                r"x",
                r"ERR named capturing group is missing trailing '>'",
            ),
            (
                r"(?<a>x)(?<a>y)",
                r"xy",
                r"ERR Named capturing group <a> is already defined",
            ),
            (r"(?<=a+)b", r"ab", r"(1,2)"),
            (r"(?$a)", r"a", r"ERR Unknown group type"),
            (r"(?@a)", r"a", r"ERR Unknown group type"),
            (r"(?i)a(?-i)a", r"AaAA", r"(0,2)"),
            (r"(?i-i)a(?is-m:.)", r"a\n", r"(0,2)"),
            (r"(?q)a", r"a", r"ERR Unknown inline modifier"),
            (r"[a[b][c]d]+", r"abcdx", r"(0,4)"),
            (r"[a-c&&b-d&&[c]]", r"abc", r"(2,3)"),
            (r"[[a]&&[b]]", r"ab", r""),
            (r"[&&ab]", r"ab", r"(0,1)(1,2)"),
            (r"[a&&]", r"a&", r"(0,1)"),
            (r"[a&&&b]", r"a&b", r"(0,1)(1,2)(2,3)"),
            (r"[ab&&", r"a", r"ERR Unclosed character class"),
            (r"[^[a]]", r"ab", r"(1,2)"),
            (r"[a&&[^b]c]", r"abc", r"(0,1)"),
            (r"[\pL\p{N}]+", r"a1-", r"(0,2)"),
            (r"[\b]", r"b", r"ERR Illegal/unsupported escape sequence"),
            (r"[\G]", r"G", r"ERR Illegal/unsupported escape sequence"),
            (r"[a-\d]", r"a", r"ERR Illegal character range"),
            (r"[a-\x{63}]+", r"abcd", r"(0,3)"),
            (r"[\x{1F600}-\x{1F601}]+", r"😀😁", r"(0,4)"),
            (r"\p{L", r"a", r"ERR Unclosed character family"),
            (r"\p{}", r"a", r"ERR Empty character family"),
            (r"\p{gc=Lu}\p{general_category=Ll}\p{IsL}", r"Abc", r"(0,3)"),
            (
                r"\p{foo=bar}",
                r"a",
                r"ERR Unknown Unicode property {name=<foo>, value=<bar>}",
            ),
            (
                r"\p{blk=Basic_Latin}\p{block=Greek}\p{sc=Grek}\p{script=Latin}",
                r"aαβb",
                r"(0,4)",
            ),
            (r"(?i)\p{IsLowercase}(?i)\p{javaLowerCase}", r"AB", r"(0,2)"),
            (r"(?iU)\p{Lower}\p{Upper}", r"Ab", r"(0,2)"),
            (
                r"\p{javaFoo}",
                r"a",
                r"ERR Unknown character property name {javaFoo}",
            ),
            (
                r"\p{IsFoo}",
                r"a",
                r"ERR Unknown character property name {IsFoo}",
            ),
            (r"(?U)\p{Alpha}\p{Punct}\p{XDigit}", r"é!f", r"(0,3)"),
            (r"a)", r"a", r"ERR Unmatched closing ')'"),
            (r"(a)\2", r"aa", r""),
            (r"\2a|a", r"a", r"(0,1)"),
            (r"(?<=😀|ab)c", r"😀c abc", r"(2,3)(6,7)"),
            (r"(?<=😀*😀*)b", r"😀b", r"(2,3)"),
            (r"(?<=😀*a*a)b", r"ab", r"(1,2)"),
            (r"(?<=a😀?)b", r"ab a😀b", r"(1,2)(6,7)"),
            (r"(?<!😀{1,2})b", r"😀😀b b", r"(6,7)"),
            (r"(?i)(ab)\1", r"abAB abA", r"(0,4 0:2)"),
            (r"(?iu)(é)\1", r"éÉ éx", r"(0,2 0:1)"),
            (r"(?i)(a)\1", r"aA ab", r"(0,2 0:1)"),
            (r"(?i)(A)\1", r"Aa", r"(0,2 0:1)"),
            (r"(a|ab){0}c", r"abc", r"(2,3 -1:-1)"),
            (r"(a|ab){0,0}?c", r"abc", r"(2,3 -1:-1)"),
            (r"(?:(a)|b)*+\1", r"aab", r""),
            (r"(?x) ( a ) \1 # c", r"aa", r"(0,2 0:1)"),
            (r"(?x)[ a b ]", r" a", r"(1,2)"),
            (r"(?d)a$", r"a\r", r""),
            (r"(?d)(?m)^a", r"\ra", r""),
            (r"(?m)$", r"a\r\nb\n", r"(1,1)(4,4)(5,5)"),
            (r"\R\R", r"\r\n\n", r"(0,3)"),
            (r"(?s).\Z", r"a\n", r"(0,1)(1,2)"),
            (r"\z|\G", r"ab", r"(0,0)(2,2)"),
            (r"(?U)\b.\B", r"é a", r""),
            (r"(?=a)*b", r"b", r"(0,1)"),
            (r"(?<=a){2}b", r"ab", r"(1,2)"),
            (r"(?>a)?b", r"ab", r"(0,2)"),
            // `X{0,1}` is `X?`: a zero-width iteration records its capture.
            (r"(\b){0,1}", r"ab", r"(0,0 0:0)(1,1 -1:-1)(2,2 2:2)"),
            (r"(\B){0,1}(a)", r"aab", r"(0,1 -1:-1 0:1)(1,2 1:1 1:2)"),
            (r"(?=(a)){0,1}?\1", r"ab", r"(0,1 0:1)"),
            (r"a\", r"a", r"ERR Unescaped trailing backslash"),
        ];
        for &(p, text, want) in cases {
            let got = finds(p, &units(text));
            match want.strip_prefix("ERR ") {
                Some(why) => assert!(got.starts_with("ERR ") && got.contains(why), "{p:?}: {got}"),
                None => assert_eq!(got, want, "{p:?} on {text:?}"),
            }
        }
    }

    /// A match whose backtracking overflows the caller's budget inside a
    /// negated lookaround or a zero-count quantifier is retried on the deep
    /// stack, not taken as the (wrong) match the overflow left behind.
    #[test]
    fn overflow_under_negation_retries() {
        let text = units(&format!("{}c", "ab".repeat(800)));
        for (p, want) in [
            (r"(?!(?:a|b)*c)", "(1601,1601)"),
            (r"(?>(?:a|b)*c)?", "(0,1601)(1601,1601)"),
            (r"((?:a|b)*c)?+", "(0,1601 0:1601)(1601,1601 -1:-1)"),
        ] {
            assert_eq!(finds(p, &text), want, "{p}");
        }
    }

    /// The caller's stack is never what a match or a parse is measured
    /// against: on a 1 MiB thread with most of it used, a deep match and a
    /// pattern nested past [`SHALLOW_NESTING`] still finish (the match on
    /// the heap, the parse and its nested sub-programs on a thread of their
    /// own).
    #[test]
    fn small_stack_caller() {
        fn burn(n: usize, f: &dyn Fn()) {
            let buf = [0u8; 1024];
            std::hint::black_box(&buf);
            if n == 0 {
                f()
            } else {
                burn(n - 1, f)
            }
        }
        let h = std::thread::Builder::new()
            .stack_size(1 << 20)
            .spawn(|| {
                // In a release build these frames are some 1 KiB each: most
                // of the 1 MiB is gone before the regex runs.
                burn(700, &|| {
                    let text = units(&format!("{}c", "ab".repeat(20_000)));
                    let prog = compile("(?=a)(?:a|b)*c|(?!(?:a|b)*c)").unwrap();
                    let mut st = State::new(&text, prog.group_total(), 0);
                    assert!(prog.search(0, &mut st).unwrap());
                    assert_eq!(st.group(0), (0, 40_001));
                    let nested = format!("{}a{}", "(?=".repeat(40), ")".repeat(40));
                    let prog = compile(&nested).unwrap();
                    let mut st = State::new(&text, prog.group_total(), 0);
                    assert!(prog.search(0, &mut st).unwrap());
                    assert_eq!(st.group(0), (0, 0));
                    let mut st = State::new(&text[..1], prog.group_total(), 0);
                    assert!(!prog.matches_all(&mut st).unwrap());
                });
            })
            .unwrap();
        assert!(h.join().is_ok());
        assert_eq!(nesting_bound(&[0x5C, 0x28, 0x28, 0x5B, 0x26, 0x61]), 3);
        // `\b` reads an ASCII unit by the ASCII rule, `(?U)` or not.
        assert!((0..128).all(|c| is_word(c, true) == is_word(c, false)));
    }

    /// Java's `StackOverflowError`: a typed error, from `find()` and
    /// `matches()` alike.
    #[test]
    fn deep_backtracking_overflows() {
        // Five entries an iteration: past `MAX_ENTRIES` at some 210,000.
        let prog = compile("(a|b)*c").unwrap();
        let text = units(&"a".repeat(300_000));
        let mut st = State::new(&text, prog.group_total(), 0);
        let e = prog.search(0, &mut st).unwrap_err();
        assert!(e.to_string().contains("StackOverflowError"), "{e}");
        // ... which is far deeper than Java's own limit (some 1,500 at
        // 1 MiB): 150,000 iterations still match.
        let mut st = State::new(&text[..150_000], prog.group_total(), 0);
        assert!(!prog.search(0, &mut st).unwrap());
        let ok = units(&format!("{}c", "ab".repeat(75_000)));
        let mut st = State::new(&ok, prog.group_total(), 0);
        assert!(prog.matches_all(&mut st).unwrap());
        assert_eq!(st.group(1), (149_999, 150_000));
        // A greedy repetition's back-off positions count too.
        let atomic = compile("(?>a|b)*c").unwrap();
        let long = units(&"a".repeat(1_100_000));
        let mut st = State::new(&long, atomic.group_total(), 0);
        assert!(atomic.search(0, &mut st).is_err());
        let mut st = State::new(&text, prog.group_total(), 0);
        assert!(prog.matches_all(&mut st).is_err());
        let mut st = State::new(&text[..10], prog.group_total(), 0);
        assert!(!prog.matches_all(&mut st).unwrap());
        assert_eq!(st.group(5), (-1, -1));
        let deep = format!("{}a{}", "(".repeat(100_000), ")".repeat(100_000));
        assert!(compile(&deep)
            .unwrap_err()
            .to_string()
            .contains("StackOverflowError"));
        let deep = format!("{}a", "[".repeat(100_000));
        assert!(compile(&deep)
            .unwrap_err()
            .to_string()
            .contains("StackOverflowError"));
        let named = compile("(?<word>a)").unwrap();
        assert_eq!(
            (named.group_named("word"), named.group_named("x")),
            (Some(1), None)
        );
    }
}
