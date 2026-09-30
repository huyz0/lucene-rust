//! `RegExp`: Lucene's regular expression syntax, parsed into a tree and
//! compiled to an [`Automaton`].
//!
//! The parser walks the pattern in UTF-16 code units, as Java's does, so
//! error positions match Lucene's messages exactly. Syntax flags
//! ([`RegExp::INTERSECTION`], [`RegExp::EMPTY`], [`RegExp::ANYSTRING`],
//! [`RegExp::AUTOMATON`], [`RegExp::INTERVAL`], [`RegExp::ALL`],
//! [`RegExp::NONE`], plus [`RegExp::DEPRECATED_COMPLEMENT`] for the pre-10
//! `~` operator) and match flags ([`RegExp::ASCII_CASE_INSENSITIVE`],
//! [`RegExp::CASE_INSENSITIVE`], [`RegExp::CASE_INSENSITIVE_RANGE`]) are
//! Lucene 10.5.0's.

use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;

use super::automata;
use super::automaton::{Automaton, Transition, TransitionAccessor};
use super::case_folding;
use super::error::AutomatonError;
use super::operations;
use super::{MAX_CODE_POINT, MIN_CODE_POINT};

/// `RegExp.Kind`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
pub enum Kind {
    /// `a|b`
    REGEXP_UNION,
    /// `ab`
    REGEXP_CONCATENATION,
    /// `a&b`
    REGEXP_INTERSECTION,
    /// `a?`
    REGEXP_OPTIONAL,
    /// `a*`
    REGEXP_REPEAT,
    /// `a{n,}`
    REGEXP_REPEAT_MIN,
    /// `a{n,m}`
    REGEXP_REPEAT_MINMAX,
    /// `[^...]`'s complement
    REGEXP_COMPLEMENT,
    /// a single character
    REGEXP_CHAR,
    /// `[a-z]`
    REGEXP_CHAR_RANGE,
    /// `[abc]`, `\d`, ...
    REGEXP_CHAR_CLASS,
    /// `.`
    REGEXP_ANYCHAR,
    /// `#`
    REGEXP_EMPTY,
    /// `"..."` or merged literals
    REGEXP_STRING,
    /// `@`
    REGEXP_ANYSTRING,
    /// `<name>`
    REGEXP_AUTOMATON,
    /// `<n-m>`
    REGEXP_INTERVAL,
    /// the deprecated `~` operator
    REGEXP_DEPRECATED_COMPLEMENT,
}

/// `AutomatonProvider`: resolves `<name>` identifiers.
pub trait AutomatonProvider {
    /// `getAutomaton(name)`; `Ok(None)` when unknown.
    ///
    /// # Errors
    /// Any lookup failure, which `RegExp` reports as `IllegalArgument`.
    fn get_automaton(&self, name: &str) -> Result<Option<Automaton>, String>;
}

/// `RegExp`: a parsed regular expression node.
#[derive(Clone, Debug)]
pub struct RegExp {
    /// Node kind.
    pub kind: Kind,
    /// First child.
    pub exp1: Option<Box<RegExp>>,
    /// Second child.
    pub exp2: Option<Box<RegExp>>,
    /// String (`REGEXP_STRING`) or identifier (`REGEXP_AUTOMATON`).
    pub s: Option<String>,
    /// Character (`REGEXP_CHAR`).
    pub c: i32,
    /// Repeat/interval minimum.
    pub min: i32,
    /// Repeat/interval maximum.
    pub max: i32,
    /// Interval digit count (0 for variable width).
    pub digits: i32,
    /// Class range starts.
    pub from: Option<Vec<i32>>,
    /// Class range ends.
    pub to: Option<Vec<i32>>,
    original_string: Option<String>,
    flags: i32,
}

struct Parser {
    units: Vec<u16>,
    pos: usize,
    flags: i32,
}

type PResult = Result<RegExp, AutomatonError>;

fn err<T>(msg: impl Into<String>) -> Result<T, AutomatonError> {
    Err(AutomatonError::IllegalArgument(msg.into()))
}

impl RegExp {
    /// Enables intersection (`&`).
    pub const INTERSECTION: i32 = 0x0001;
    /// Enables the empty language (`#`).
    pub const EMPTY: i32 = 0x0004;
    /// Enables any string (`@`).
    pub const ANYSTRING: i32 = 0x0008;
    /// Enables named automata (`<identifier>`).
    pub const AUTOMATON: i32 = 0x0010;
    /// Enables numerical intervals (`<n-m>`).
    pub const INTERVAL: i32 = 0x0020;
    /// All optional syntax.
    pub const ALL: i32 = 0xff;
    /// No optional syntax.
    pub const NONE: i32 = 0x0000;
    /// Deprecated ASCII-only case-insensitive matching (now full Unicode).
    pub const ASCII_CASE_INSENSITIVE: i32 = 0x0100;
    /// Case-insensitive matching of literal characters.
    pub const CASE_INSENSITIVE: i32 = 0x0200;
    /// Case-insensitive matching of character class ranges too.
    pub const CASE_INSENSITIVE_RANGE: i32 = 0x0400;
    /// The deprecated `~` complement operator.
    pub const DEPRECATED_COMPLEMENT: i32 = 0x10000;

    /// `new RegExp(s)`: with [`RegExp::ALL`] syntax.
    ///
    /// # Errors
    /// `IllegalArgument` on a syntax error.
    pub fn new(s: &str) -> PResult {
        Self::with_flags(s, Self::ALL, 0)
    }

    /// `new RegExp(s, syntaxFlags, matchFlags)`.
    ///
    /// # Errors
    /// `IllegalArgument` on a syntax error or an illegal flag.
    pub fn with_flags(s: &str, syntax_flags: i32, match_flags: i32) -> PResult {
        if (syntax_flags & !Self::DEPRECATED_COMPLEMENT) > Self::ALL {
            return err("Illegal syntax flag");
        }
        if match_flags > 0 && match_flags <= Self::ALL {
            return err("Illegal match flag");
        }
        let flags = syntax_flags | match_flags;
        let mut p = Parser {
            units: s.encode_utf16().collect(),
            pos: 0,
            flags,
        };
        let mut e = if s.is_empty() {
            make_string(flags, String::new())
        } else {
            let e = p.parse_union_exp()?;
            if p.pos < p.units.len() {
                return err(format!("end-of-string expected at position {}", p.pos));
            }
            e
        };
        e.original_string = Some(s.to_string());
        e.flags = flags;
        Ok(e)
    }

    fn node(flags: i32, kind: Kind) -> RegExp {
        RegExp {
            kind,
            exp1: None,
            exp2: None,
            s: None,
            c: 0,
            min: 0,
            max: 0,
            digits: 0,
            from: None,
            to: None,
            original_string: None,
            flags,
        }
    }

    fn exp1(&self) -> &RegExp {
        self.exp1.as_deref().expect("container node has exp1")
    }

    fn exp2(&self) -> &RegExp {
        self.exp2.as_deref().expect("binary node has exp2")
    }

    fn check(&self, flag: i32) -> bool {
        self.flags & flag != 0
    }

    /// `getOriginalString()`: the pattern this was parsed from (top-level
    /// node only).
    pub fn get_original_string(&self) -> Option<&str> {
        self.original_string.as_deref()
    }

    /// `toAutomaton()`.
    ///
    /// # Errors
    /// `IllegalArgument` for an unknown `<name>`, or when a decimal interval
    /// is invalid.
    pub fn to_automaton(&self) -> Result<Automaton, AutomatonError> {
        self.to_automaton_internal(None, None)
    }

    /// `toAutomaton(AutomatonProvider)`.
    ///
    /// # Errors
    /// As [`RegExp::to_automaton`].
    pub fn to_automaton_with_provider(
        &self,
        provider: &dyn AutomatonProvider,
    ) -> Result<Automaton, AutomatonError> {
        self.to_automaton_internal(None, Some(provider))
    }

    /// `toAutomaton(Map<String, Automaton>)`.
    ///
    /// # Errors
    /// As [`RegExp::to_automaton`].
    pub fn to_automaton_with_map(
        &self,
        automata: &HashMap<String, Automaton>,
    ) -> Result<Automaton, AutomatonError> {
        self.to_automaton_internal(Some(automata), None)
    }

    fn to_automaton_internal(
        &self,
        map: Option<&HashMap<String, Automaton>>,
        provider: Option<&dyn AutomatonProvider>,
    ) -> Result<Automaton, AutomatonError> {
        Ok(match self.kind {
            Kind::REGEXP_UNION => {
                let mut list = Vec::new();
                find_leaves(self.exp1(), Kind::REGEXP_UNION, &mut list, map, provider)?;
                find_leaves(self.exp2(), Kind::REGEXP_UNION, &mut list, map, provider)?;
                let refs: Vec<&Automaton> = list.iter().collect();
                operations::union(&refs)
            }
            Kind::REGEXP_CONCATENATION => {
                let mut list = Vec::new();
                find_leaves(
                    self.exp1(),
                    Kind::REGEXP_CONCATENATION,
                    &mut list,
                    map,
                    provider,
                )?;
                find_leaves(
                    self.exp2(),
                    Kind::REGEXP_CONCATENATION,
                    &mut list,
                    map,
                    provider,
                )?;
                let refs: Vec<&Automaton> = list.iter().collect();
                operations::concatenate(&refs)
            }
            Kind::REGEXP_INTERSECTION => operations::intersection(
                &self.exp1().to_automaton_internal(map, provider)?,
                &self.exp2().to_automaton_internal(map, provider)?,
            ),
            Kind::REGEXP_OPTIONAL => {
                operations::optional(&self.exp1().to_automaton_internal(map, provider)?)
            }
            Kind::REGEXP_REPEAT => {
                operations::repeat(&self.exp1().to_automaton_internal(map, provider)?)
            }
            Kind::REGEXP_REPEAT_MIN => {
                operations::repeat_min(&self.exp1().to_automaton_internal(map, provider)?, self.min)
            }
            Kind::REGEXP_REPEAT_MINMAX => operations::repeat_range(
                &self.exp1().to_automaton_internal(map, provider)?,
                self.min,
                self.max,
            ),
            Kind::REGEXP_COMPLEMENT => operations::complement(
                &self.exp1().to_automaton_internal(map, provider)?,
                i32::MAX,
            )?,
            Kind::REGEXP_DEPRECATED_COMPLEMENT => operations::complement(
                &self.exp1().to_automaton_internal(map, provider)?,
                operations::DEFAULT_DETERMINIZE_WORK_LIMIT,
            )?,
            Kind::REGEXP_CHAR => {
                if self.check(Self::ASCII_CASE_INSENSITIVE | Self::CASE_INSENSITIVE) {
                    automata::make_case_insensitive_char(self.c)
                } else {
                    automata::make_char(self.c)
                }
            }
            Kind::REGEXP_CHAR_RANGE => {
                let (f, t) = self.ranges();
                automata::make_char_range(f[0], t[0])
            }
            Kind::REGEXP_CHAR_CLASS => {
                let (f, t) = self.ranges();
                automata::make_char_class(f, t)?
            }
            Kind::REGEXP_ANYCHAR => automata::make_any_char(),
            Kind::REGEXP_EMPTY => automata::make_empty(),
            Kind::REGEXP_STRING => {
                let s = self.s.as_deref().unwrap_or("");
                if self.check(Self::ASCII_CASE_INSENSITIVE | Self::CASE_INSENSITIVE) {
                    automata::make_case_insensitive_string(s)
                } else {
                    automata::make_string(s)
                }
            }
            Kind::REGEXP_ANYSTRING => automata::make_any_string(),
            Kind::REGEXP_AUTOMATON => {
                let name = self.s.as_deref().unwrap_or("");
                let mut aa = map.and_then(|m| m.get(name).cloned());
                if aa.is_none() {
                    if let Some(p) = provider {
                        aa = p
                            .get_automaton(name)
                            .map_err(AutomatonError::IllegalArgument)?;
                    }
                }
                match aa {
                    Some(a) => a,
                    None => return err(format!("'{name}' not found")),
                }
            }
            Kind::REGEXP_INTERVAL => {
                automata::make_decimal_interval(self.min, self.max, self.digits)?
            }
        })
    }

    fn ranges(&self) -> (&[i32], &[i32]) {
        (
            self.from.as_deref().expect("class node has from"),
            self.to.as_deref().expect("class node has to"),
        )
    }

    /// `toStringTree()`: one line per node, children indented.
    pub fn to_string_tree(&self) -> String {
        let mut b = String::new();
        self.to_string_tree_into(&mut b, "");
        b
    }

    fn to_string_tree_into(&self, b: &mut String, indent: &str) {
        let child_indent = format!("{indent}  ");
        match self.kind {
            Kind::REGEXP_UNION | Kind::REGEXP_CONCATENATION | Kind::REGEXP_INTERSECTION => {
                let _ = writeln!(b, "{indent}{:?}", self.kind);
                self.exp1().to_string_tree_into(b, &child_indent);
                self.exp2().to_string_tree_into(b, &child_indent);
            }
            Kind::REGEXP_OPTIONAL
            | Kind::REGEXP_REPEAT
            | Kind::REGEXP_COMPLEMENT
            | Kind::REGEXP_DEPRECATED_COMPLEMENT => {
                let _ = writeln!(b, "{indent}{:?}", self.kind);
                self.exp1().to_string_tree_into(b, &child_indent);
            }
            Kind::REGEXP_REPEAT_MIN => {
                let _ = writeln!(b, "{indent}{:?} min={}", self.kind, self.min);
                self.exp1().to_string_tree_into(b, &child_indent);
            }
            Kind::REGEXP_REPEAT_MINMAX => {
                let _ = writeln!(
                    b,
                    "{indent}{:?} min={} max={}",
                    self.kind, self.min, self.max
                );
                self.exp1().to_string_tree_into(b, &child_indent);
            }
            Kind::REGEXP_CHAR => {
                let _ = write!(b, "{indent}{:?} char=", self.kind);
                append_code_point(b, self.c);
                b.push('\n');
            }
            Kind::REGEXP_CHAR_RANGE => {
                let (f, t) = self.ranges();
                let _ = write!(b, "{indent}{:?} from=", self.kind);
                append_code_point(b, f[0]);
                b.push_str(" to=");
                append_code_point(b, t[0]);
                b.push('\n');
            }
            Kind::REGEXP_CHAR_CLASS => {
                let (f, t) = self.ranges();
                let _ = writeln!(
                    b,
                    "{indent}{:?} starts={} ends={}",
                    self.kind,
                    hex_list(f),
                    hex_list(t)
                );
            }
            Kind::REGEXP_ANYCHAR
            | Kind::REGEXP_EMPTY
            | Kind::REGEXP_ANYSTRING
            | Kind::REGEXP_AUTOMATON => {
                let _ = writeln!(b, "{indent}{:?}", self.kind);
            }
            Kind::REGEXP_STRING => {
                let _ = writeln!(
                    b,
                    "{indent}{:?} string={}",
                    self.kind,
                    self.s.as_deref().unwrap_or("")
                );
            }
            Kind::REGEXP_INTERVAL => {
                let _ = write!(b, "{indent}{:?}", self.kind);
                self.append_interval(b);
                b.push('\n');
            }
        }
    }

    fn append_interval(&self, b: &mut String) {
        let s1 = self.min.to_string();
        let s2 = self.max.to_string();
        b.push('<');
        if self.digits > 0 {
            for _ in s1.len()..self.digits as usize {
                b.push('0');
            }
        }
        b.push_str(&s1);
        b.push('-');
        if self.digits > 0 {
            for _ in s2.len()..self.digits as usize {
                b.push('0');
            }
        }
        b.push_str(&s2);
        b.push('>');
    }

    fn to_string_into(&self, b: &mut String) {
        match self.kind {
            Kind::REGEXP_UNION => {
                b.push('(');
                self.exp1().to_string_into(b);
                b.push('|');
                self.exp2().to_string_into(b);
                b.push(')');
            }
            Kind::REGEXP_CONCATENATION => {
                self.exp1().to_string_into(b);
                self.exp2().to_string_into(b);
            }
            Kind::REGEXP_INTERSECTION => {
                b.push('(');
                self.exp1().to_string_into(b);
                b.push('&');
                self.exp2().to_string_into(b);
                b.push(')');
            }
            Kind::REGEXP_OPTIONAL => {
                b.push('(');
                self.exp1().to_string_into(b);
                b.push_str(")?");
            }
            Kind::REGEXP_REPEAT => {
                b.push('(');
                self.exp1().to_string_into(b);
                b.push_str(")*");
            }
            Kind::REGEXP_REPEAT_MIN => {
                b.push('(');
                self.exp1().to_string_into(b);
                let _ = write!(b, "){{{},}}", self.min);
            }
            Kind::REGEXP_REPEAT_MINMAX => {
                b.push('(');
                self.exp1().to_string_into(b);
                let _ = write!(b, "){{{},{}}}", self.min, self.max);
            }
            Kind::REGEXP_COMPLEMENT | Kind::REGEXP_DEPRECATED_COMPLEMENT => {
                b.push_str("~(");
                self.exp1().to_string_into(b);
                b.push(')');
            }
            Kind::REGEXP_CHAR => {
                b.push('\\');
                append_code_point(b, self.c);
            }
            Kind::REGEXP_CHAR_RANGE => {
                let (f, t) = self.ranges();
                b.push_str("[\\");
                append_code_point(b, f[0]);
                b.push_str("-\\");
                append_code_point(b, t[0]);
                b.push(']');
            }
            Kind::REGEXP_CHAR_CLASS => {
                let (f, t) = self.ranges();
                b.push('[');
                for (&lo, &hi) in f.iter().zip(t) {
                    b.push('\\');
                    append_code_point(b, lo);
                    if lo != hi {
                        b.push_str("-\\");
                        append_code_point(b, hi);
                    }
                }
                b.push(']');
            }
            Kind::REGEXP_ANYCHAR => b.push('.'),
            Kind::REGEXP_EMPTY => b.push('#'),
            Kind::REGEXP_STRING => {
                b.push('"');
                b.push_str(self.s.as_deref().unwrap_or(""));
                b.push('"');
            }
            Kind::REGEXP_ANYSTRING => b.push('@'),
            Kind::REGEXP_AUTOMATON => {
                b.push('<');
                b.push_str(self.s.as_deref().unwrap_or(""));
                b.push('>');
            }
            Kind::REGEXP_INTERVAL => self.append_interval(b),
        }
    }

    /// `getIdentifiers()`: every `<name>` in the expression.
    pub fn get_identifiers(&self) -> HashSet<String> {
        let mut set = HashSet::new();
        self.collect_identifiers(&mut set);
        set
    }

    fn collect_identifiers(&self, set: &mut HashSet<String>) {
        match self.kind {
            Kind::REGEXP_UNION | Kind::REGEXP_CONCATENATION | Kind::REGEXP_INTERSECTION => {
                self.exp1().collect_identifiers(set);
                self.exp2().collect_identifiers(set);
            }
            Kind::REGEXP_OPTIONAL
            | Kind::REGEXP_REPEAT
            | Kind::REGEXP_REPEAT_MIN
            | Kind::REGEXP_REPEAT_MINMAX
            | Kind::REGEXP_COMPLEMENT
            | Kind::REGEXP_DEPRECATED_COMPLEMENT => self.exp1().collect_identifiers(set),
            Kind::REGEXP_AUTOMATON => {
                set.insert(self.s.clone().unwrap_or_default());
            }
            _ => {}
        }
    }
}

impl std::fmt::Display for RegExp {
    /// Lucene's `RegExp.toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut b = String::new();
        self.to_string_into(&mut b);
        f.write_str(&b)
    }
}

/// `StringBuilder.appendCodePoint`; a surrogate code point (only reachable
/// through a class range bound) becomes U+FFFD, as Rust strings cannot hold
/// one.
fn append_code_point(b: &mut String, c: i32) {
    b.push(char::from_u32(c as u32).unwrap_or('\u{FFFD}'));
}

fn hex_list(v: &[i32]) -> String {
    let parts: Vec<String> = v.iter().map(|c| format!("U+{c:04X}")).collect();
    format!("[{}]", parts.join(" "))
}

fn find_leaves(
    exp: &RegExp,
    kind: Kind,
    list: &mut Vec<Automaton>,
    map: Option<&HashMap<String, Automaton>>,
    provider: Option<&dyn AutomatonProvider>,
) -> Result<(), AutomatonError> {
    if exp.kind == kind {
        find_leaves(exp.exp1(), kind, list, map, provider)?;
        find_leaves(exp.exp2(), kind, list, map, provider)?;
    } else {
        list.push(exp.to_automaton_internal(map, provider)?);
    }
    Ok(())
}

// --- node constructors (Java's static make* helpers) -----------------------

fn container(flags: i32, kind: Kind, e1: RegExp, e2: Option<RegExp>) -> RegExp {
    let mut r = RegExp::node(flags, kind);
    r.exp1 = Some(Box::new(e1));
    r.exp2 = e2.map(Box::new);
    r
}

fn is_char_or_string(e: &RegExp) -> bool {
    e.kind == Kind::REGEXP_CHAR || e.kind == Kind::REGEXP_STRING
}

fn make_union(flags: i32, e1: RegExp, e2: RegExp) -> RegExp {
    container(flags, Kind::REGEXP_UNION, e1, Some(e2))
}

fn make_concatenation(flags: i32, e1: RegExp, e2: RegExp) -> RegExp {
    if is_char_or_string(&e1) && is_char_or_string(&e2) {
        return make_string_of(flags, &e1, &e2);
    }
    let (r1, r2) = if e1.kind == Kind::REGEXP_CONCATENATION
        && is_char_or_string(e1.exp2())
        && is_char_or_string(&e2)
    {
        let RegExp { exp1, exp2, .. } = e1;
        let (a, b) = (*exp1.expect("exp1"), *exp2.expect("exp2"));
        (a, make_string_of(flags, &b, &e2))
    } else if is_char_or_string(&e1)
        && e2.kind == Kind::REGEXP_CONCATENATION
        && is_char_or_string(e2.exp1())
    {
        let RegExp { exp1, exp2, .. } = e2;
        let (a, b) = (*exp1.expect("exp1"), *exp2.expect("exp2"));
        (make_string_of(flags, &e1, &a), b)
    } else {
        (e1, e2)
    };
    container(flags, Kind::REGEXP_CONCATENATION, r1, Some(r2))
}

fn make_string_of(flags: i32, e1: &RegExp, e2: &RegExp) -> RegExp {
    let mut b = String::new();
    for e in [e1, e2] {
        if e.kind == Kind::REGEXP_STRING {
            b.push_str(e.s.as_deref().unwrap_or(""));
        } else {
            append_code_point(&mut b, e.c);
        }
    }
    make_string(flags, b)
}

fn make_intersection(flags: i32, e1: RegExp, e2: RegExp) -> RegExp {
    container(flags, Kind::REGEXP_INTERSECTION, e1, Some(e2))
}

fn make_char(flags: i32, c: i32) -> RegExp {
    let mut r = RegExp::node(flags, Kind::REGEXP_CHAR);
    r.c = c;
    r
}

fn make_char_range(flags: i32, from: i32, to: i32) -> PResult {
    if from > to {
        return err(format!(
            "invalid range: from ({from}) cannot be > to ({to})"
        ));
    }
    let mut r = RegExp::node(flags, Kind::REGEXP_CHAR_RANGE);
    r.from = Some(vec![from]);
    r.to = Some(vec![to]);
    Ok(r)
}

fn make_char_class(flags: i32, from: Vec<i32>, to: Vec<i32>) -> PResult {
    for (&f, &t) in from.iter().zip(&to) {
        if f > t {
            return err(format!("invalid range: from ({f}) cannot be > to ({t})"));
        }
    }
    let mut r = RegExp::node(flags, Kind::REGEXP_CHAR_CLASS);
    r.from = Some(from);
    r.to = Some(to);
    Ok(r)
}

fn make_string(flags: i32, s: String) -> RegExp {
    let mut r = RegExp::node(flags, Kind::REGEXP_STRING);
    r.s = Some(s);
    r
}

// --- the parser -------------------------------------------------------------

impl Parser {
    fn check(&self, flag: i32) -> bool {
        self.flags & flag != 0
    }

    fn more(&self) -> bool {
        self.pos < self.units.len()
    }

    /// `String.codePointAt(pos)`.
    fn code_point_at(&self, pos: usize) -> i32 {
        let hi = u32::from(self.units[pos]);
        if (0xD800..0xDC00).contains(&hi) && pos + 1 < self.units.len() {
            let lo = u32::from(self.units[pos + 1]);
            if (0xDC00..0xE000).contains(&lo) {
                return (((hi - 0xD800) << 10) + (lo - 0xDC00) + 0x10000) as i32;
            }
        }
        hi as i32
    }

    fn char_count(c: i32) -> usize {
        if c >= 0x10000 {
            2
        } else {
            1
        }
    }

    fn peek(&self, s: &str) -> bool {
        self.more() && {
            let c = self.code_point_at(self.pos);
            s.chars().any(|x| x as i32 == c)
        }
    }

    fn match_char(&mut self, c: char) -> bool {
        if self.pos >= self.units.len() {
            return false;
        }
        if self.code_point_at(self.pos) == c as i32 {
            self.pos += Self::char_count(c as i32);
            return true;
        }
        false
    }

    fn next(&mut self) -> Result<i32, AutomatonError> {
        if !self.more() {
            return err("unexpected end-of-string");
        }
        let ch = self.code_point_at(self.pos);
        self.pos += Self::char_count(ch);
        Ok(ch)
    }

    fn substring(&self, start: usize, end: usize) -> String {
        String::from_utf16_lossy(&self.units[start..end])
    }

    fn parse_union_exp(&mut self) -> PResult {
        let mut result = self.parse_inter_exp()?;
        while self.match_char('|') {
            let e = self.parse_inter_exp()?;
            result = make_union(self.flags, result, e);
        }
        Ok(result)
    }

    fn parse_inter_exp(&mut self) -> PResult {
        let mut result = self.parse_concat_exp()?;
        while self.check(RegExp::INTERSECTION) && self.match_char('&') {
            let e = self.parse_concat_exp()?;
            result = make_intersection(self.flags, result, e);
        }
        Ok(result)
    }

    fn parse_concat_exp(&mut self) -> PResult {
        let mut result = self.parse_repeat_exp()?;
        while self.more()
            && !self.peek(")|")
            && (!self.check(RegExp::INTERSECTION) || !self.peek("&"))
        {
            let e = self.parse_repeat_exp()?;
            result = make_concatenation(self.flags, result, e);
        }
        Ok(result)
    }

    fn parse_int(&self, start: usize, end: usize) -> Result<i32, AutomatonError> {
        let s = self.substring(start, end);
        s.parse::<i32>()
            .map_err(|_| AutomatonError::IllegalArgument(format!("For input string: \"{s}\"")))
    }

    fn parse_repeat_exp(&mut self) -> PResult {
        let mut e = self.parse_compl_exp()?;
        let flags = self.flags;
        while self.peek("?*+{") {
            if self.match_char('?') {
                e = container(flags, Kind::REGEXP_OPTIONAL, e, None);
            } else if self.match_char('*') {
                e = container(flags, Kind::REGEXP_REPEAT, e, None);
            } else if self.match_char('+') {
                let mut r = container(flags, Kind::REGEXP_REPEAT_MIN, e, None);
                r.min = 1;
                e = r;
            } else if self.match_char('{') {
                let mut start = self.pos;
                while self.peek("0123456789") {
                    self.next()?;
                }
                if start == self.pos {
                    return err(format!("integer expected at position {}", self.pos));
                }
                let n = self.parse_int(start, self.pos)?;
                let mut m = -1;
                if self.match_char(',') {
                    start = self.pos;
                    while self.peek("0123456789") {
                        self.next()?;
                    }
                    if start != self.pos {
                        m = self.parse_int(start, self.pos)?;
                    }
                } else {
                    m = n;
                }
                if !self.match_char('}') {
                    return err(format!("expected '}}' at position {}", self.pos));
                }
                if m != -1 && n > m {
                    return err(format!("invalid repetition range(out of order): {n}..{m}"));
                }
                if m == -1 {
                    let mut r = container(flags, Kind::REGEXP_REPEAT_MIN, e, None);
                    r.min = n;
                    e = r;
                } else {
                    let mut r = container(flags, Kind::REGEXP_REPEAT_MINMAX, e, None);
                    r.min = n;
                    r.max = m;
                    e = r;
                }
            }
        }
        Ok(e)
    }

    fn parse_compl_exp(&mut self) -> PResult {
        if self.check(RegExp::DEPRECATED_COMPLEMENT) && self.match_char('~') {
            let inner = self.parse_compl_exp()?;
            Ok(container(
                self.flags,
                Kind::REGEXP_DEPRECATED_COMPLEMENT,
                inner,
                None,
            ))
        } else {
            self.parse_char_class_exp()
        }
    }

    fn parse_char_class_exp(&mut self) -> PResult {
        if self.match_char('[') {
            let negate = self.match_char('^');
            let mut e = self.parse_char_classes()?;
            if negate {
                let flags = self.flags;
                e = make_intersection(
                    flags,
                    RegExp::node(flags, Kind::REGEXP_ANYCHAR),
                    container(flags, Kind::REGEXP_COMPLEMENT, e, None),
                );
            }
            if !self.match_char(']') {
                return err(format!("expected ']' at position {}", self.pos));
            }
            Ok(e)
        } else {
            self.parse_simple_exp()
        }
    }

    fn parse_char_classes(&mut self) -> PResult {
        let mut starts: Vec<i32> = Vec::new();
        let mut ends: Vec<i32> = Vec::new();
        loop {
            if self.match_char('\\') {
                if self.peek(PREDEFINED) {
                    self.expand_pre_defined(&mut starts, &mut ends)?;
                } else {
                    let c = self.next()?;
                    starts.push(c);
                    ends.push(c);
                }
            } else {
                let c = self.parse_char_exp()?;
                if self.match_char('-') {
                    if self.check(RegExp::CASE_INSENSITIVE_RANGE) {
                        let to = self.parse_char_exp()?;
                        expand_case_insensitive_range(c, to, &mut starts, &mut ends)?;
                    } else {
                        starts.push(c);
                        ends.push(self.parse_char_exp()?);
                    }
                } else if self.check(RegExp::ASCII_CASE_INSENSITIVE | RegExp::CASE_INSENSITIVE) {
                    for form in automata::to_case_insensitive_char(c) {
                        starts.push(form);
                        ends.push(form);
                    }
                } else {
                    starts.push(c);
                    ends.push(c);
                }
            }
            if !(self.more() && !self.peek("]")) {
                break;
            }
        }
        if starts.len() == 1 {
            if starts[0] == ends[0] {
                Ok(make_char(self.flags, starts[0]))
            } else {
                make_char_range(self.flags, starts[0], ends[0])
            }
        } else {
            make_char_class(self.flags, starts, ends)
        }
    }

    fn expand_pre_defined(
        &mut self,
        starts: &mut Vec<i32>,
        ends: &mut Vec<i32>,
    ) -> Result<(), AutomatonError> {
        let add = |s: &mut Vec<i32>, e: &mut Vec<i32>, pairs: &[(i32, i32)]| {
            for &(a, b) in pairs {
                s.push(a);
                e.push(b);
            }
        };
        let c = |ch: char| ch as i32;
        if self.peek("\\") {
            add(starts, ends, &[(c('\\'), c('\\'))]);
            self.next()?;
        } else if self.peek("d") {
            add(starts, ends, &[(c('0'), c('9'))]);
            self.next()?;
        } else if self.peek("D") {
            add(
                starts,
                ends,
                &[(MIN_CODE_POINT, c('0') - 1), (c('9') + 1, MAX_CODE_POINT)],
            );
            self.next()?;
        } else if self.peek("s") {
            add(
                starts,
                ends,
                &[(c('\t'), c('\n')), (c('\r'), c('\r')), (c(' '), c(' '))],
            );
            self.next()?;
        } else if self.peek("S") {
            add(
                starts,
                ends,
                &[
                    (MIN_CODE_POINT, c('\t') - 1),
                    (c('\n') + 1, c('\r') - 1),
                    (c('\r') + 1, c(' ') - 1),
                    (c(' ') + 1, MAX_CODE_POINT),
                ],
            );
            self.next()?;
        } else if self.peek("w") {
            add(
                starts,
                ends,
                &[
                    (c('0'), c('9')),
                    (c('A'), c('Z')),
                    (c('_'), c('_')),
                    (c('a'), c('z')),
                ],
            );
            self.next()?;
        } else if self.peek("W") {
            add(
                starts,
                ends,
                &[
                    (MIN_CODE_POINT, c('0') - 1),
                    (c('9') + 1, c('A') - 1),
                    (c('Z') + 1, c('_') - 1),
                    (c('_') + 1, c('a') - 1),
                    (c('z') + 1, MAX_CODE_POINT),
                ],
            );
            self.next()?;
        } else if self.peek("abcefghijklmnopqrtuvxyz") || self.peek("ABCEFGHIJKLMNOPQRTUVXYZ") {
            let ch = self.next()?;
            let mut m = String::from("invalid character class \\");
            // Java appends the int returned by next(): its decimal value.
            let _ = write!(m, "{ch}");
            return err(m);
        }
        Ok(())
    }

    fn match_predefined_character_class(&mut self) -> Result<Option<RegExp>, AutomatonError> {
        if self.match_char('\\') && self.peek(PREDEFINED) {
            let mut starts = Vec::new();
            let mut ends = Vec::new();
            self.expand_pre_defined(&mut starts, &mut ends)?;
            return make_char_class(self.flags, starts, ends).map(Some);
        }
        Ok(None)
    }

    fn parse_simple_exp(&mut self) -> PResult {
        let flags = self.flags;
        if self.match_char('.') {
            Ok(RegExp::node(flags, Kind::REGEXP_ANYCHAR))
        } else if self.check(RegExp::EMPTY) && self.match_char('#') {
            Ok(RegExp::node(flags, Kind::REGEXP_EMPTY))
        } else if self.check(RegExp::ANYSTRING) && self.match_char('@') {
            Ok(RegExp::node(flags, Kind::REGEXP_ANYSTRING))
        } else if self.match_char('"') {
            let start = self.pos;
            while self.more() && !self.peek("\"") {
                self.next()?;
            }
            if !self.match_char('"') {
                return err(format!("expected '\"' at position {}", self.pos));
            }
            Ok(make_string(flags, self.substring(start, self.pos - 1)))
        } else if self.match_char('(') {
            if self.match_char(')') {
                return Ok(make_string(flags, String::new()));
            }
            let e = self.parse_union_exp()?;
            if !self.match_char(')') {
                return err(format!("expected ')' at position {}", self.pos));
            }
            Ok(e)
        } else if (self.check(RegExp::AUTOMATON) || self.check(RegExp::INTERVAL))
            && self.match_char('<')
        {
            let start = self.pos;
            while self.more() && !self.peek(">") {
                self.next()?;
            }
            if !self.match_char('>') {
                return err(format!("expected '>' at position {}", self.pos));
            }
            let s = self.substring(start, self.pos - 1);
            let at = self.pos - 1;
            match s.find('-') {
                None => {
                    if !self.check(RegExp::AUTOMATON) {
                        return err(format!("interval syntax error at position {at}"));
                    }
                    let mut r = RegExp::node(flags, Kind::REGEXP_AUTOMATON);
                    r.s = Some(s);
                    Ok(r)
                }
                Some(i) => {
                    if !self.check(RegExp::INTERVAL) {
                        return err(format!("illegal identifier at position {at}"));
                    }
                    let syntax = || {
                        AutomatonError::IllegalArgument(format!(
                            "interval syntax error at position {at}"
                        ))
                    };
                    if i == 0 || i == s.len() - 1 || Some(i) != s.rfind('-') {
                        return Err(syntax());
                    }
                    let smin = &s[..i];
                    let smax = &s[i + 1..];
                    let mut imin = smin.parse::<i32>().map_err(|_| syntax())?;
                    let mut imax = smax.parse::<i32>().map_err(|_| syntax())?;
                    let digits = if smin.encode_utf16().count() == smax.encode_utf16().count() {
                        smin.encode_utf16().count() as i32
                    } else {
                        0
                    };
                    if imin > imax {
                        std::mem::swap(&mut imin, &mut imax);
                    }
                    let mut r = RegExp::node(flags, Kind::REGEXP_INTERVAL);
                    r.min = imin;
                    r.max = imax;
                    r.digits = digits;
                    Ok(r)
                }
            }
        } else {
            if let Some(p) = self.match_predefined_character_class()? {
                return Ok(p);
            }
            let c = self.parse_char_exp()?;
            Ok(make_char(flags, c))
        }
    }

    fn parse_char_exp(&mut self) -> Result<i32, AutomatonError> {
        self.match_char('\\');
        self.next()
    }
}

const PREDEFINED: &str = "\\ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

/// Java's `expandCaseInsensitiveRange`: every case variant of every code
/// point in `start..=end`, as the merged ranges of a one-state automaton.
fn expand_case_insensitive_range(
    start: i32,
    end: i32,
    starts: &mut Vec<i32>,
    ends: &mut Vec<i32>,
) -> Result<(), AutomatonError> {
    if start > end {
        return err(format!(
            "invalid range: from ({start}) cannot be > to ({end})"
        ));
    }
    let mut scratch = Automaton::new();
    let state = scratch.create_state();
    for i in start..=end {
        case_folding::expand(i, &mut |ch| scratch.add_transition(state, state, ch, ch));
    }
    scratch.finish_state();
    let mut t = Transition::new();
    let n = scratch.init_transition(state, &mut t);
    for _ in 0..n {
        scratch.get_next_transition(&mut t);
        starts.push(t.min);
        ends.push(t.max);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::automaton::operations::{determinize, run};

    fn accepts(re: &str, flags: i32, mflags: i32, s: &str) -> bool {
        let a = RegExp::with_flags(re, flags, mflags)
            .unwrap()
            .to_automaton()
            .unwrap();
        run(&determinize(&a, 100_000).unwrap(), s)
    }

    #[test]
    fn syntax() {
        assert!(accepts("ab+c", RegExp::ALL, 0, "abbbc"));
        assert!(accepts("[^a-c]x", RegExp::ALL, 0, "dx"));
        assert!(!accepts("[^a-c]x", RegExp::ALL, 0, "bx"));
        assert!(accepts("<1-100>", RegExp::ALL, 0, "0042"));
        assert!(accepts("<01-10>", RegExp::ALL, 0, "07"));
        assert!(!accepts("<01-10>", RegExp::ALL, 0, "7"));
        assert!(accepts(
            "@&~(a.*)",
            RegExp::ALL | RegExp::DEPRECATED_COMPLEMENT,
            0,
            "ba"
        ));
        assert!(accepts("~a", RegExp::ALL, 0, "~a"));
        assert!(accepts("\\d{2,3}", RegExp::ALL, 0, "123"));
        assert!(accepts("x{2,}", RegExp::ALL, 0, "xxxx"));
        assert!(accepts(
            "QUICK",
            RegExp::ALL,
            RegExp::CASE_INSENSITIVE,
            "quick"
        ));
        assert!(accepts(
            "[a-c]",
            RegExp::ALL,
            RegExp::CASE_INSENSITIVE_RANGE,
            "B"
        ));
        assert!(accepts(
            "[k]",
            RegExp::ALL,
            RegExp::CASE_INSENSITIVE,
            "\u{212A}"
        ));
        assert!(accepts("", RegExp::ALL, 0, ""));
        assert!(accepts("()", RegExp::ALL, 0, ""));
        assert!(accepts("\"a.b\"", RegExp::ALL, 0, "a.b"));
        assert!(accepts("\\w\\W\\s\\S\\D", RegExp::ALL, 0, "a! xy"));
        assert!(accepts("[\\d\\\\]", RegExp::ALL, 0, "\\"));
    }

    #[test]
    fn errors() {
        for (re, msg) in [
            ("a{", "integer expected at position 2"),
            ("a{2", "expected '}' at position 3"),
            ("a{3,2}", "invalid repetition range(out of order): 3..2"),
            ("[a", "expected ']' at position 2"),
            ("(a", "expected ')' at position 2"),
            ("\"a", "expected '\"' at position 2"),
            ("<a", "expected '>' at position 2"),
            ("<-1>", "interval syntax error at position 3"),
            ("a)", "end-of-string expected at position 1"),
            ("[z-a]", "invalid range: from (122) cannot be > to (97)"),
            ("\\p", "invalid character class \\112"),
            ("a\\", "unexpected end-of-string"),
        ] {
            let e = RegExp::new(re).unwrap_err();
            assert_eq!(e, AutomatonError::IllegalArgument(msg.into()), "{re}");
        }
        assert!(RegExp::with_flags("a", 0x1000, 0).is_err());
        assert!(RegExp::with_flags("a", RegExp::ALL, 0x10).is_err());
        assert_eq!(
            RegExp::with_flags("<a-b>", RegExp::AUTOMATON, 0).unwrap_err(),
            AutomatonError::IllegalArgument("illegal identifier at position 4".into())
        );
        assert_eq!(
            RegExp::with_flags("<1-2>", RegExp::INTERVAL, 0)
                .unwrap()
                .kind,
            Kind::REGEXP_INTERVAL
        );
        assert_eq!(
            RegExp::with_flags("<ab>", RegExp::INTERVAL, 0).unwrap_err(),
            AutomatonError::IllegalArgument("interval syntax error at position 3".into())
        );
        assert!(RegExp::new("a{99999999999}").is_err());
        assert!(RegExp::new("<name>").unwrap().to_automaton().is_err());
    }

    struct Provider;
    impl AutomatonProvider for Provider {
        fn get_automaton(&self, name: &str) -> Result<Option<Automaton>, String> {
            match name {
                "abc" => Ok(Some(automata::make_string("abc"))),
                "boom" => Err("io".into()),
                _ => Ok(None),
            }
        }
    }

    #[test]
    fn named_automata_and_printing() {
        let r = RegExp::new("x<abc>").unwrap();
        let a = r.to_automaton_with_provider(&Provider).unwrap();
        assert!(run(&a, "xabc"));
        assert!(RegExp::new("<boom>")
            .unwrap()
            .to_automaton_with_provider(&Provider)
            .is_err());
        assert!(RegExp::new("<nope>")
            .unwrap()
            .to_automaton_with_provider(&Provider)
            .is_err());
        let mut m = HashMap::new();
        m.insert("abc".to_string(), automata::make_string("q"));
        assert!(run(&r.to_automaton_with_map(&m).unwrap(), "xq"));
        assert_eq!(
            r.get_identifiers().into_iter().collect::<Vec<_>>(),
            vec!["abc"]
        );
        assert_eq!(r.get_original_string(), Some("x<abc>"));
        let r = RegExp::with_flags(
            "(a|b)&c?d*e+f{2}g{3,}h{4,5}[i-k][lm].#@\"n\"<007-009>~o[^p]",
            RegExp::ALL | RegExp::DEPRECATED_COMPLEMENT,
            0,
        )
        .unwrap();
        let s = r.to_string();
        assert!(s.contains("(\\a|\\b)"), "{s}");
        assert!(
            s.contains("[\\i-\\k]") && s.contains("<007-009>") && s.contains("~(\\o)"),
            "{s}"
        );
        let tree = r.to_string_tree();
        assert!(
            tree.starts_with("REGEXP_INTERSECTION\n  REGEXP_UNION\n"),
            "{tree}"
        );
        assert!(tree.contains("REGEXP_CHAR_CLASS starts=[U+006C U+006D] ends=[U+006C U+006D]"));
        assert!(tree.contains("REGEXP_REPEAT_MINMAX min=4 max=5"));
        assert!(tree.contains("REGEXP_INTERVAL<007-009>"));
        assert!(tree.contains("REGEXP_DEPRECATED_COMPLEMENT"));
    }
}
