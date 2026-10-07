//! `java.util.regex.Pattern`/`Matcher`, as the analysis-common `pattern`
//! package and `PatternKeywordMarkerFilter` use them, over the `regex`
//! crate.
//!
//! A Java pattern is parsed with `regex-syntax`'s parser (after a pre-pass
//! for the Java-only escapes `\Q..\E`, `\h`, `\H`, `\v`, `\V`, `\e` and
//! `\uD8xx\uDCxx` pairs) and **re-emitted**: every character class, literal
//! and `.` becomes an explicit code point set computed with Java's rules,
//! so no construct reaches the `regex` crate with a meaning of its own.
//! What is Java's here:
//!
//! - **Classes.** `\d`, `\w`, `\s` and the POSIX names (`\p{Alpha}`,
//!   `\p{Punct}`, `\p{Upper}`, ...) are ASCII; general categories
//!   (`\p{L}`, `\p{IsLu}`, `\p{gc=Nd}`, `\pN`, `LC`, `LD`, `L1`, `all`) come
//!   from [`java_character_tables`](crate::java_character_tables) (JDK 25's
//!   `Character.getType`). `[[:alpha:]]` is Java's nested class of the
//!   characters `:alph`, not a POSIX class. `\h` and `\v` are Java's
//!   whitespace classes (`\v` is not U+000B).
//! - **Case-insensitivity.** `(?i)` alone folds ASCII only; `(?iu)` folds
//!   with `Character.toUpperCase`/`toLowerCase` (JDK 25 simple mappings,
//!   `lucene_util`'s table), with Java's own asymmetries: a single literal
//!   whose upper and lower forms agree matches only itself (`ß` does not
//!   match `ẞ`) while the same character inside a run of literals does;
//!   a class literal below U+0100 adds its upper and lower forms only;
//!   a range matches `ch` when `ch`, its uppercase or the lowercase of its
//!   uppercase is in range; `\p{Lu}`, `\p{Ll}`, `\p{Lt}` widen to all cased
//!   letters and `\p{Upper}`/`\p{Lower}` to `\p{Alpha}`.
//! - **`.`** excludes `\n`, `\r`, U+0085, U+2028 and U+2029 (all of them
//!   with `(?s)`).
//! - **`$`** matches at the end and before a final line terminator (`\n`
//!   not after `\r`, `\r\n`, `\r`, U+0085, U+2028, U+2029). The `regex`
//!   crate has no lookahead, so a text ending in a terminator is searched
//!   with a sentinel byte (`0xFF`, never valid UTF-8) inserted before the
//!   terminator: `$` is `(?:\z|\xFF)` and every other atom may step over
//!   the sentinel, so it stands for the zero-width position exactly.
//! - **`find()`** resumes where the last match ended, and one UTF-16 unit
//!   on after an empty match. After an empty match before a supplementary
//!   character that lands *between its surrogates*, where Java tries the
//!   pattern once more against the lone low surrogate: the port replays
//!   that attempt with a marker byte for the low half that exactly the
//!   classes containing the surrogates accept. A match boundary inside a
//!   pair is reported in UTF-16 units like Java's; text cut there holds
//!   U+FFFD for the half in a `String` (the crate's convention, and what
//!   Java's indexing turns a lone surrogate into) and the unit itself in
//!   UTF-16 output ([`JavaMatcher::slice_utf16_into`]).
//! - **Replacement strings** are `appendReplacement`'s: `$n` (greedy over
//!   digits while the group exists), `${name}`, `\` quoting.
//! - Matching itself is leftmost-first over code points, which is what
//!   Java's backtracking computes for the constructs that remain.
//!
//! **Handed to the backtracking matcher** ([`super::java_backtrack`]),
//! which has `Pattern`'s own semantics, is every pattern the shim refuses:
//! anything `regex-syntax` cannot parse (backreferences, lookaround,
//! possessive and atomic groups, `\G`, `\Z`, `\R`, `\cX`, octal `\0n`, lone
//! surrogate escapes); `\b` and `\B` (Java's boundary counts a non-spacing
//! mark after a letter or digit as a word character); `^`/`$` under `(?m)`;
//! more than one `$` at a position; the flags `(?x)`, `(?U)` and `(?d)`;
//! script, block and binary properties and the `java*` properties; a
//! quantifier on a quantifier; `~~` and an empty or `&`-led `&&` operand; a
//! repeated group holding a capture, whose captures Java reports otherwise
//! (`(a|)*`, `((a)|b)*`). The shim's [`UNSUPPORTED`] refusal reaches the
//! caller only for what the backtracking matcher refuses too (`\X`,
//! `\b{g}`, `\N{..}`, `(?c)`); any other error the shim reports is
//! replaced by the backtracking parser's, which follows Java's grammar, so
//! a pattern Java rejects is rejected with Java's
//! `PatternSyntaxException:` description (`(?R)`, `(?P<..>)`, `\u{..}`,
//! `x{ 2 }`, `--` among them, which the `regex` crate alone would take).

use std::borrow::Cow;
use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use regex::bytes::{CaptureLocations, Regex};
use regex_syntax::ast::{
    self, AssertionKind, Ast, ClassPerlKind, ClassSet, ClassSetBinaryOpKind, ClassSetItem,
    ClassUnicodeKind, ClassUnicodeOpKind, Flag, FlagsItemKind, GroupKind, HexLiteralKind,
    LiteralKind, RepetitionKind, RepetitionRange,
};

use super::java_backtrack::{self, Program, Scratch, State};
use crate::java_character::{to_lower_case, to_upper_case};
use crate::java_character_tables::GENERAL_CATEGORY_RUNS;
use crate::AnalysisError;

const MAX_CP: u32 = 0x10_FFFF;
/// Inserted before a final line terminator so `$` can see it.
const SENTINEL: u8 = 0xFF;
/// Stands for a lone low surrogate in a mid-pair attempt.
const LOW_HALF: u8 = 0xFD;
/// Precedes [`LOW_HALF`], so `\A` is false where it stands.
const PAD: u8 = 0xFE;
/// Java's `\h`.
const H_CLASS: &str =
    r"[\x{20}\x{9}\x{A0}\x{1680}\x{180E}\x{2000}-\x{200A}\x{202F}\x{205F}\x{3000}]";
/// Java's `\H`.
const NOT_H_CLASS: &str =
    r"[^\x{20}\x{9}\x{A0}\x{1680}\x{180E}\x{2000}-\x{200A}\x{202F}\x{205F}\x{3000}]";
/// Java's `\v`.
const V_CLASS: &str = r"[\x{A}-\x{D}\x{85}\x{2028}\x{2029}]";
/// Java's `\V`.
const NOT_V_CLASS: &str = r"[^\x{A}-\x{D}\x{85}\x{2028}\x{2029}]";

fn syntax(e: impl std::fmt::Display) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("PatternSyntaxException: {e}"))
}

/// A construct Java refuses too (a `PatternSyntaxException` there), which
/// the `regex` crate would have accepted.
fn java_rejects(what: &str, why: &str) -> AnalysisError {
    syntax(format!("`{what}`: {why}"))
}

/// How [`JavaPattern::compile`]'s message starts when the shim refuses a
/// pattern Java compiles (see [`is_unsupported`]).
pub const UNSUPPORTED: &str = "unsupported java.util.regex construct";

fn unsupported(what: &str, why: &str) -> AnalysisError {
    AnalysisError::IllegalArgument(format!("{UNSUPPORTED} `{what}`: {why}"))
}

/// Whether [`JavaPattern::compile`] refused a pattern that Java compiles
/// (a limit of this shim), rather than one Java refuses too.
pub fn is_unsupported(e: &AnalysisError) -> bool {
    matches!(e, AnalysisError::IllegalArgument(m) if m.starts_with(UNSUPPORTED))
}

/// The first construct of `src` that Java compiles and `regex-syntax`
/// cannot parse -- consulted only once the parse has failed, so a pattern
/// that is also malformed elsewhere is reported as unsupported.
fn java_only_construct(src: &str) -> Option<(String, &'static str)> {
    let chars: Vec<char> = src.chars().collect();
    let mut depth = 0usize;
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        let rest: String = chars[i..chars.len().min(i + 4)].iter().collect();
        if c == '\\' {
            let Some(&e) = chars.get(i + 1) else { break };
            let surrogate = e == 'u'
                && chars.get(i + 2).is_some_and(|c| matches!(c, 'd' | 'D'))
                && chars
                    .get(i + 3)
                    .is_some_and(|c| matches!(c, '8'..='9' | 'a'..='f' | 'A'..='F'));
            let why = match e {
                _ if surrogate => Some("lone surrogate escapes"),
                '1'..='9' | 'k' => Some("backreferences"),
                'G' | 'Z' | 'R' | 'X' | 'N' | 'c' | '0' => {
                    Some("an escape the regex crate does not have")
                }
                _ => None,
            };
            if let Some(why) = why {
                return Some((format!("\\{e}"), why));
            }
            i += 2;
            continue;
        }
        if c == '[' {
            depth += 1;
        } else if c == ']' && depth > 0 {
            depth -= 1;
        } else if depth == 0 {
            for (lead, why) in [
                ("(?<=", "lookaround"),
                ("(?<!", "lookaround"),
                ("(?=", "lookaround"),
                ("(?!", "lookaround"),
                ("(?>", "atomic groups"),
            ] {
                if rest.starts_with(lead) {
                    return Some((lead.to_string(), why));
                }
            }
            if matches!(c, '*' | '+' | '?' | '}') && chars.get(i + 1) == Some(&'+') {
                return Some((format!("{c}+"), "possessive quantifiers"));
            }
        }
        i += 1;
    }
    None
}

/// A `regex-syntax` parse error: unsupported when the pattern holds a
/// construct only Java has, else Java's syntax error too.
fn parse_error(src: &str, e: ast::Error) -> AnalysisError {
    match e.kind() {
        ast::ErrorKind::UnsupportedBackreference => unsupported(src, "backreferences"),
        ast::ErrorKind::UnsupportedLookAround => unsupported(src, "lookaround"),
        _ => match java_only_construct(src) {
            Some((what, why)) => unsupported(&what, why),
            None => syntax(e),
        },
    }
}

// ------------------------------------------------------------ code point sets

/// A set of code points (surrogates included, as Java's classes hold them):
/// sorted, disjoint, non-adjacent inclusive ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(super) struct CpSet(pub(super) Vec<(u32, u32)>);

impl CpSet {
    pub(super) fn range(a: u32, b: u32) -> Self {
        CpSet(vec![(a, b)])
    }

    pub(super) fn single(c: u32) -> Self {
        CpSet::range(c, c)
    }

    pub(super) fn all() -> Self {
        CpSet::range(0, MAX_CP)
    }

    pub(super) fn from_points(points: impl IntoIterator<Item = u32>) -> Self {
        CpSet::from_ranges(points.into_iter().map(|c| (c, c)).collect())
    }

    pub(super) fn from_ranges(mut v: Vec<(u32, u32)>) -> Self {
        v.sort_unstable();
        let mut out: Vec<(u32, u32)> = Vec::with_capacity(v.len());
        for (a, b) in v {
            match out.last_mut() {
                Some(last) if a <= last.1.saturating_add(1) => last.1 = last.1.max(b),
                _ => out.push((a, b)),
            }
        }
        CpSet(out)
    }

    pub(super) fn union(&self, o: &CpSet) -> CpSet {
        let mut v = self.0.clone();
        v.extend_from_slice(&o.0);
        CpSet::from_ranges(v)
    }

    pub(super) fn intersect(&self, o: &CpSet) -> CpSet {
        let (mut i, mut j) = (0, 0);
        let mut out = Vec::new();
        while i < self.0.len() && j < o.0.len() {
            let (a, b) = self.0[i];
            let (c, d) = o.0[j];
            let (lo, hi) = (a.max(c), b.min(d));
            if lo <= hi {
                out.push((lo, hi));
            }
            if b < d {
                i += 1;
            } else {
                j += 1;
            }
        }
        CpSet(out)
    }

    pub(super) fn complement(&self) -> CpSet {
        let mut out = Vec::new();
        let mut next = 0u32;
        for &(a, b) in &self.0 {
            if a > next {
                out.push((next, a - 1));
            }
            next = b + 1;
        }
        if next <= MAX_CP {
            out.push((next, MAX_CP));
        }
        CpSet(out)
    }

    pub(super) fn contains(&self, c: u32) -> bool {
        let i = self.0.partition_point(|&(_, b)| b < c);
        i < self.0.len() && self.0[i].0 <= c
    }
}

/// Java's `Character.getType` bit for each category letter pair.
fn category_mask(name: &str) -> Option<u32> {
    const LU: u32 = 1 << 1;
    const LL: u32 = 1 << 2;
    const LT: u32 = 1 << 3;
    const LM: u32 = 1 << 4;
    const LO: u32 = 1 << 5;
    const MN: u32 = 1 << 6;
    const ME: u32 = 1 << 7;
    const MC: u32 = 1 << 8;
    const ND: u32 = 1 << 9;
    const NL: u32 = 1 << 10;
    const NO: u32 = 1 << 11;
    const ZS: u32 = 1 << 12;
    const ZL: u32 = 1 << 13;
    const ZP: u32 = 1 << 14;
    const CC: u32 = 1 << 15;
    const CF: u32 = 1 << 16;
    const CO: u32 = 1 << 18;
    const CS: u32 = 1 << 19;
    const PD: u32 = 1 << 20;
    const PS: u32 = 1 << 21;
    const PE: u32 = 1 << 22;
    const PC: u32 = 1 << 23;
    const PO: u32 = 1 << 24;
    const SM: u32 = 1 << 25;
    const SC: u32 = 1 << 26;
    const SK: u32 = 1 << 27;
    const SO: u32 = 1 << 28;
    const PI: u32 = 1 << 29;
    const PF: u32 = 1 << 30;
    const CN: u32 = 1;
    Some(match name {
        "Cn" => CN,
        "Lu" => LU,
        "Ll" => LL,
        "Lt" => LT,
        "Lm" => LM,
        "Lo" => LO,
        "Mn" => MN,
        "Me" => ME,
        "Mc" => MC,
        "Nd" => ND,
        "Nl" => NL,
        "No" => NO,
        "Zs" => ZS,
        "Zl" => ZL,
        "Zp" => ZP,
        "Cc" => CC,
        "Cf" => CF,
        "Co" => CO,
        "Cs" => CS,
        "Pd" => PD,
        "Ps" => PS,
        "Pe" => PE,
        "Pc" => PC,
        "Po" => PO,
        "Sm" => SM,
        "Sc" => SC,
        "Sk" => SK,
        "So" => SO,
        "Pi" => PI,
        "Pf" => PF,
        "L" => LU | LL | LT | LM | LO,
        "M" => MN | ME | MC,
        "N" => ND | NL | NO,
        "Z" => ZS | ZL | ZP,
        "C" => CC | CF | CO | CS | CN,
        "P" => PD | PS | PE | PC | PO | PI | PF,
        "S" => SM | SC | SK | SO,
        "LC" => LU | LL | LT,
        "LD" => LU | LL | LT | LM | LO | ND,
        _ => return None,
    })
}

/// The code points whose `Character.getType` is in `mask`.
fn category(mask: u32) -> CpSet {
    let runs = &GENERAL_CATEGORY_RUNS;
    let mut v = Vec::new();
    for (i, &(start, t)) in runs.iter().enumerate() {
        if mask & (1 << t) != 0 {
            let end = runs.get(i + 1).map_or(MAX_CP, |&(next, _)| next - 1);
            v.push((start, end));
        }
    }
    CpSet::from_ranges(v)
}

/// `ASCII.toUpper`/`toLower`.
fn ascii_upper(c: u32) -> u32 {
    if (u32::from(b'a')..=u32::from(b'z')).contains(&c) {
        c - 0x20
    } else {
        c
    }
}

fn ascii_lower(c: u32) -> u32 {
    if (u32::from(b'A')..=u32::from(b'Z')).contains(&c) {
        c + 0x20
    } else {
        c
    }
}

fn ascii_set(spec: &[(u8, u8)]) -> CpSet {
    CpSet::from_ranges(spec.iter().map(|&(a, b)| (a.into(), b.into())).collect())
}

/// `CharPredicates.forProperty(name, caseIns)`: the categories and the
/// ASCII POSIX classes (the `java*` properties are not supported).
pub(super) fn for_property(name: &str, ci: bool) -> Option<CpSet> {
    const ALPHA: [(u8, u8); 2] = [(b'A', b'Z'), (b'a', b'z')];
    Some(match name {
        "Lu" | "Ll" | "Lt" if ci => category(category_mask("LC")?),
        "L1" => CpSet::range(0, 0xFF),
        "all" => CpSet::all(),
        "ASCII" => CpSet::range(0, 0x7F),
        "Alnum" => ascii_set(&[(b'0', b'9'), (b'A', b'Z'), (b'a', b'z')]),
        "Alpha" => ascii_set(&ALPHA),
        "Blank" => ascii_set(&[(b' ', b' '), (b'\t', b'\t')]),
        "Cntrl" => ascii_set(&[(0, 0x1F), (0x7F, 0x7F)]),
        "Digit" => ascii_set(&[(b'0', b'9')]),
        "Graph" => ascii_set(&[(0x21, 0x7E)]),
        "Lower" if ci => ascii_set(&ALPHA),
        "Lower" => ascii_set(&[(b'a', b'z')]),
        "Print" => ascii_set(&[(0x20, 0x7E)]),
        "Punct" => ascii_set(&[(0x21, 0x2F), (0x3A, 0x40), (0x5B, 0x60), (0x7B, 0x7E)]),
        "Space" => ascii_set(&[(b' ', b' '), (0x09, 0x0D)]),
        "Upper" if ci => ascii_set(&ALPHA),
        "Upper" => ascii_set(&[(b'A', b'Z')]),
        "XDigit" => ascii_set(&[(b'0', b'9'), (b'A', b'F'), (b'a', b'f')]),
        _ => category(category_mask(name)?),
    })
}

/// `\d`, `\s`, `\w`: Java's ASCII classes.
fn perl(kind: &ClassPerlKind) -> CpSet {
    match kind {
        ClassPerlKind::Digit => ascii_set(&[(b'0', b'9')]),
        ClassPerlKind::Space => ascii_set(&[(b' ', b' '), (0x09, 0x0D)]),
        ClassPerlKind::Word => ascii_set(&[(b'0', b'9'), (b'A', b'Z'), (b'_', b'_'), (b'a', b'z')]),
    }
}

// ------------------------------------------------------- case-insensitivity

/// Every code point with a non-identity simple case mapping, with its
/// `toUpperCase` and `toLowerCase(toUpperCase)`, and the inverse of the
/// latter. Simple case mappings stop below U+20000 (checked by a test).
struct CaseIndex {
    cased: Vec<(u32, u32, u32)>,
    by_lower_upper: HashMap<u32, Vec<u32>>,
}

/// Upper bound (exclusive) of the code points with a simple case mapping.
const CASED_LIMIT: u32 = 0x2_0000;

static CASE_INDEX: LazyLock<CaseIndex> = LazyLock::new(|| {
    let mut cased = Vec::new();
    let mut by_lower_upper: HashMap<u32, Vec<u32>> = HashMap::new();
    for c in 0..CASED_LIMIT {
        let (u, l) = (to_upper_case(c), to_lower_case(c));
        if u == c && l == c {
            continue;
        }
        let lu = to_lower_case(u);
        cased.push((c, u, lu));
        by_lower_upper.entry(lu).or_default().push(c);
    }
    CaseIndex {
        cased,
        by_lower_upper,
    }
});

/// `Character.toLowerCase(Character.toUpperCase(c))`.
fn lower_upper(c: u32) -> u32 {
    to_lower_case(to_upper_case(c))
}

/// What a literal `c` matches outside a class: Java's `single(c)` (a lone
/// literal) or one character of a `Slice` (a run of literals).
pub(super) fn literal_set(c: u32, f: JFlags, in_run: bool) -> CpSet {
    if !f.ci {
        return CpSet::single(c);
    }
    if !f.uc {
        return CpSet::from_points([c, ascii_lower(c), ascii_upper(c)]);
    }
    let lower = lower_upper(c);
    // `single` matches only `c` when its upper and lower forms agree;
    // `SliceU` compares `toLowerCase(toUpperCase(..))` regardless.
    if !in_run && to_upper_case(c) == lower {
        return CpSet::single(c);
    }
    unicode_class_of(lower)
}

/// `{x} ∪ {ch : toLowerCase(toUpperCase(ch)) == x}`.
fn unicode_class_of(x: u32) -> CpSet {
    let mut pts = vec![x];
    if let Some(v) = CASE_INDEX.by_lower_upper.get(&x) {
        pts.extend_from_slice(v);
    }
    CpSet::from_points(pts)
}

/// A literal inside a class: `bitsOrSingle`.
pub(super) fn class_literal_set(c: u32, f: JFlags) -> CpSet {
    const UNICODE_SINGLE: [u32; 10] = [0xFF, 0xB5, 0x49, 0x69, 0x53, 0x73, 0x4B, 0x6B, 0xC5, 0xE5];
    if c < 0x100 && !(f.ci && f.uc && UNICODE_SINGLE.contains(&c)) {
        // BitClass.add
        if !f.ci {
            return CpSet::single(c);
        }
        if c < 0x80 {
            return CpSet::from_points([c, ascii_upper(c), ascii_lower(c)]);
        }
        if f.uc {
            return CpSet::from_points([c, to_lower_case(c), to_upper_case(c)]);
        }
        return CpSet::single(c);
    }
    literal_set(c, f, false)
}

/// A range inside a class: `Range`, `CIRange` or `CIRangeU`.
pub(super) fn class_range_set(a: u32, b: u32, f: JFlags) -> CpSet {
    let base = CpSet::range(a, b);
    if !f.ci {
        return base;
    }
    let mut extra = Vec::new();
    if f.uc {
        for &(ch, up, lu) in &CASE_INDEX.cased {
            if (a..=b).contains(&up) || (a..=b).contains(&lu) {
                extra.push((ch, ch));
            }
        }
    } else {
        for ch in 0..0x80u32 {
            if (a..=b).contains(&ascii_upper(ch)) || (a..=b).contains(&ascii_lower(ch)) {
                extra.push((ch, ch));
            }
        }
    }
    base.union(&CpSet::from_ranges(extra))
}

// --------------------------------------------------------------- the pre-pass

/// Rewrites the Java-only escapes into syntax `regex-syntax` parses with
/// the same meaning: `\Q..\E` quoting, `\h \H \v \V \e`, and a surrogate
/// pair spelled as two `\u` escapes.
fn prepass(p: &str) -> Cow<'_, str> {
    if !p.contains('\\') {
        return Cow::Borrowed(p);
    }
    let chars: Vec<char> = p.chars().collect();
    let mut out = String::with_capacity(p.len() + 16);
    let hex4 = |i: usize| -> Option<u32> {
        let s: String = chars.get(i..i + 4)?.iter().collect();
        if s.chars().all(|c| c.is_ascii_hexdigit()) {
            u32::from_str_radix(&s, 16).ok()
        } else {
            None
        }
    };
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c != '\\' || i + 1 == chars.len() {
            out.push(c);
            i += 1;
            continue;
        }
        let e = chars[i + 1];
        i += 2;
        match e {
            'Q' => {
                while i < chars.len() {
                    if chars[i] == '\\' && chars.get(i + 1) == Some(&'E') {
                        i += 2;
                        break;
                    }
                    out.push_str(&format!("\\x{{{:X}}}", u32::from(chars[i])));
                    i += 1;
                }
            }
            'h' => out.push_str(H_CLASS),
            'H' => out.push_str(NOT_H_CLASS),
            'v' => out.push_str(V_CLASS),
            'V' => out.push_str(NOT_V_CLASS),
            'e' => out.push_str(r"\x{1B}"),
            'u' => match (hex4(i), chars.get(i + 4..i + 6), hex4(i + 6)) {
                (Some(hi @ 0xD800..=0xDBFF), Some(['\\', 'u']), Some(lo @ 0xDC00..=0xDFFF)) => {
                    let cp = 0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00);
                    out.push_str(&format!("\\x{{{cp:X}}}"));
                    i += 10;
                }
                _ => out.push_str("\\u"),
            },
            _ => {
                out.push('\\');
                out.push(e);
            }
        }
    }
    Cow::Owned(out)
}

// ------------------------------------------------------------ the translator

/// Java's inline flags that the translation honours.
#[derive(Debug, Clone, Copy, Default)]
pub(super) struct JFlags {
    /// `CASE_INSENSITIVE` (`i`).
    pub(super) ci: bool,
    /// `UNICODE_CASE` (`u`).
    pub(super) uc: bool,
    /// `DOTALL` (`s`).
    pub(super) dotall: bool,
    /// `MULTILINE` (`m`).
    pub(super) multiline: bool,
}

/// One emission of a parsed Java pattern as a `regex` pattern.
struct Translator<'s> {
    src: &'s str,
    /// Atoms may step over the `$` sentinel, and `$` is `(?:\z|\xFF)`.
    sentinel: bool,
    /// Classes holding the low surrogates also accept [`LOW_HALF`].
    mid: bool,
    out: String,
    has_dollar: bool,
    has_low_class: bool,
}

impl Translator<'_> {
    fn text(&self, span: &ast::Span) -> &str {
        &self.src[span.start.offset..span.end.offset]
    }

    fn atom_open(&mut self) {
        if self.sentinel {
            self.out.push_str(r"(?:(?-u:\xFF)?");
        }
    }

    fn atom_close(&mut self) {
        if self.sentinel {
            self.out.push(')');
        }
    }

    fn emit_set(&mut self, set: &CpSet) {
        let low = set.contains(0xDC00);
        self.has_low_class |= low;
        let mut class = String::from("[");
        for &(a, b) in &set.0 {
            // A Rust class cannot hold a surrogate; no UTF-8 text has one.
            let mut parts = Vec::with_capacity(2);
            if a < 0xD800 {
                parts.push((a, b.min(0xD7FF)));
            }
            if b > 0xDFFF {
                parts.push((a.max(0xE000), b));
            }
            for (x, y) in parts {
                if x == y {
                    class.push_str(&format!("\\x{{{x:X}}}"));
                } else {
                    class.push_str(&format!("\\x{{{x:X}}}-\\x{{{y:X}}}"));
                }
            }
        }
        if class.len() == 1 {
            class.push_str(r"^\x{0}-\x{10FFFF}");
        }
        class.push(']');
        self.atom_open();
        if self.mid && low {
            self.out.push_str("(?:");
            self.out.push_str(&class);
            self.out.push_str(r"|(?-u:\xFD))");
        } else {
            self.out.push_str(&class);
        }
        self.atom_close();
    }

    fn emit_literal(&mut self, c: u32, f: JFlags, in_run: bool) {
        if f.ci {
            self.emit_set(&literal_set(c, f, in_run));
        } else {
            self.atom_open();
            self.out.push_str(&format!("\\x{{{c:X}}}"));
            self.atom_close();
        }
    }

    /// The character a literal stands for, or why Java reads it otherwise.
    fn literal_char(&self, lit: &ast::Literal) -> Result<u32, AnalysisError> {
        let bad = |why| Err(unsupported(self.text(&lit.span), why));
        match &lit.kind {
            LiteralKind::HexFixed(HexLiteralKind::UnicodeLong) => {
                Err(java_rejects(self.text(&lit.span), "Java has no \\U escape"))
            }
            LiteralKind::HexBrace(HexLiteralKind::UnicodeShort | HexLiteralKind::UnicodeLong) => {
                Err(java_rejects(
                    self.text(&lit.span),
                    "Java's \\u takes exactly four hex digits",
                ))
            }
            LiteralKind::Special(ast::SpecialLiteralKind::VerticalTab) => {
                bad("Java's \\v is a class")
            }
            _ => Ok(u32::from(lit.c)),
        }
    }

    fn apply_flags(&self, flags: &ast::Flags, f: &mut JFlags) -> Result<(), AnalysisError> {
        let mut on = true;
        for item in &flags.items {
            match &item.kind {
                FlagsItemKind::Negation => on = false,
                FlagsItemKind::Flag(flag) => match flag {
                    Flag::CaseInsensitive => f.ci = on,
                    Flag::Unicode => f.uc = on,
                    Flag::DotMatchesNewLine => f.dotall = on,
                    Flag::MultiLine => f.multiline = on,
                    Flag::SwapGreed => {
                        return Err(unsupported(
                            self.text(&flags.span),
                            "Java's (?U) is UNICODE_CHARACTER_CLASS",
                        ))
                    }
                    Flag::IgnoreWhitespace => {
                        return Err(unsupported(self.text(&flags.span), "COMMENTS mode"))
                    }
                    Flag::CRLF => {
                        return Err(java_rejects(self.text(&flags.span), "Java has no (?R)"))
                    }
                },
            }
        }
        Ok(())
    }

    fn unicode_class(&self, c: &ast::ClassUnicode, f: JFlags) -> Result<CpSet, AnalysisError> {
        let what = self.text(&c.span);
        let set = match &c.kind {
            ClassUnicodeKind::OneLetter(l) => for_property(&l.to_string(), f.ci),
            ClassUnicodeKind::Named(name) => match name.strip_prefix("In") {
                Some(_) => None,
                None => match name.strip_prefix("Is") {
                    // `Is` + a category; Java's binary properties and
                    // scripts under `Is` are not supported.
                    Some(rest) => category_mask(rest).and_then(|_| for_property(rest, f.ci)),
                    None => for_property(name, f.ci),
                },
            },
            ClassUnicodeKind::NamedValue { op, name, value } => {
                let gc = matches!(
                    name.to_ascii_lowercase().as_str(),
                    "gc" | "general_category"
                );
                if *op == ClassUnicodeOpKind::Equal && gc {
                    for_property(value, f.ci)
                } else {
                    None
                }
            }
        };
        let set = set.ok_or_else(|| {
            unsupported(
                what,
                "only general categories and the ASCII POSIX classes are supported",
            )
        })?;
        Ok(if c.negated { set.complement() } else { set })
    }

    fn class_set(&self, set: &ClassSet, f: JFlags) -> Result<CpSet, AnalysisError> {
        match set {
            ClassSet::Item(item) => self.class_item(item, f),
            ClassSet::BinaryOp(op) => {
                match op.kind {
                    ClassSetBinaryOpKind::Intersection => {}
                    // Java refuses `--`; it reads `~~` as two literals.
                    ClassSetBinaryOpKind::Difference => {
                        return Err(java_rejects(
                            self.text(&op.span),
                            "Java has no class difference operator",
                        ))
                    }
                    ClassSetBinaryOpKind::SymmetricDifference => {
                        return Err(unsupported(
                            self.text(&op.span),
                            "Java reads `~~` as literals, the regex crate as an operator",
                        ))
                    }
                }
                for side in [&op.lhs, &op.rhs] {
                    if operand_is_quirky(side) {
                        return Err(unsupported(
                            self.text(&op.span),
                            "Java reads an empty or `&`-led `&&` operand differently",
                        ));
                    }
                }
                Ok(self
                    .class_set(&op.lhs, f)?
                    .intersect(&self.class_set(&op.rhs, f)?))
            }
        }
    }

    fn class_item(&self, item: &ClassSetItem, f: JFlags) -> Result<CpSet, AnalysisError> {
        Ok(match item {
            ClassSetItem::Empty(_) => CpSet::default(),
            ClassSetItem::Literal(lit) => class_literal_set(self.literal_char(lit)?, f),
            ClassSetItem::Range(r) => {
                let (a, b) = (self.literal_char(&r.start)?, self.literal_char(&r.end)?);
                class_range_set(a, b, f)
            }
            ClassSetItem::Ascii(a) => {
                // Java: a nested class of the characters between the brackets.
                let text = self.text(&a.span);
                let inner = &text[1..text.len() - 1];
                let mut set = CpSet::default();
                for c in inner.chars() {
                    set = set.union(&class_literal_set(u32::from(c), f));
                }
                set
            }
            ClassSetItem::Unicode(u) => self.unicode_class(u, f)?,
            ClassSetItem::Perl(p) => {
                let s = perl(&p.kind);
                if p.negated {
                    s.complement()
                } else {
                    s
                }
            }
            ClassSetItem::Bracketed(b) => {
                let s = self.class_set(&b.kind, f)?;
                if b.negated {
                    s.complement()
                } else {
                    s
                }
            }
            ClassSetItem::Union(u) => {
                let mut set = CpSet::default();
                for i in &u.items {
                    set = set.union(&self.class_item(i, f)?);
                }
                set
            }
        })
    }

    fn check_counted(&self, op: &ast::RepetitionOp) -> Result<(), AnalysisError> {
        if !matches!(op.kind, RepetitionKind::Range(_)) {
            return Ok(());
        }
        let t = self.text(&op.span);
        let t = t.strip_suffix('?').unwrap_or(t);
        let body = t.strip_prefix('{').and_then(|t| t.strip_suffix('}'));
        let ok = body.is_some_and(|b| {
            let mut parts = b.splitn(2, ',');
            let lo = parts.next().unwrap_or("");
            let hi = parts.next().unwrap_or("");
            !lo.is_empty()
                && lo.bytes().all(|x| x.is_ascii_digit())
                && hi.bytes().all(|x| x.is_ascii_digit())
        });
        if ok {
            Ok(())
        } else {
            Err(java_rejects(
                t,
                "Java's counted repetition is `{n}`, `{n,}` or `{n,m}`",
            ))
        }
    }

    fn walk(&mut self, a: &Ast, f: &mut JFlags) -> Result<(), AnalysisError> {
        match a {
            Ast::Empty(_) => {}
            Ast::Flags(set) => self.apply_flags(&set.flags, f)?,
            Ast::Literal(lit) => {
                let c = self.literal_char(lit)?;
                self.emit_literal(c, *f, false);
            }
            Ast::Dot(_) => {
                let set = if f.dotall {
                    CpSet::all()
                } else {
                    CpSet::from_points([0x0A, 0x0D, 0x85, 0x2028, 0x2029]).complement()
                };
                self.emit_set(&set);
            }
            Ast::Assertion(x) => self.assertion(x, *f)?,
            Ast::ClassUnicode(c) => {
                let s = self.unicode_class(c, *f)?;
                self.emit_set(&s);
            }
            Ast::ClassPerl(p) => {
                let s = perl(&p.kind);
                self.emit_set(&if p.negated { s.complement() } else { s });
            }
            Ast::ClassBracketed(b) => {
                let s = self.class_set(&b.kind, *f)?;
                self.emit_set(&if b.negated { s.complement() } else { s });
            }
            Ast::Repetition(r) => {
                self.check_counted(&r.op)?;
                if matches!(*r.ast, Ast::Repetition(_)) {
                    // Java: `a*+` is possessive, `a**` a repeated repetition.
                    return Err(unsupported(
                        self.text(&r.span),
                        "possessive quantifiers and a quantifier on a quantifier",
                    ));
                }
                let twice = match &r.op.kind {
                    RepetitionKind::ZeroOrOne => false,
                    RepetitionKind::ZeroOrMore | RepetitionKind::OneOrMore => true,
                    RepetitionKind::Range(RepetitionRange::Exactly(n)) => *n >= 2,
                    RepetitionKind::Range(RepetitionRange::AtLeast(_)) => true,
                    RepetitionKind::Range(RepetitionRange::Bounded(_, m)) => *m >= 2,
                };
                if twice && nullable(&r.ast) && has_capture(&r.ast) {
                    return Err(unsupported(
                        self.text(&r.span),
                        "Java reports a different capture after an empty iteration",
                    ));
                }
                self.out.push_str("(?:");
                self.walk(&r.ast, f)?;
                self.out.push(')');
                let op = match &r.op.kind {
                    RepetitionKind::ZeroOrOne => "?".to_string(),
                    RepetitionKind::ZeroOrMore => "*".to_string(),
                    RepetitionKind::OneOrMore => "+".to_string(),
                    RepetitionKind::Range(RepetitionRange::Exactly(n)) => format!("{{{n}}}"),
                    RepetitionKind::Range(RepetitionRange::AtLeast(n)) => format!("{{{n},}}"),
                    RepetitionKind::Range(RepetitionRange::Bounded(n, m)) => format!("{{{n},{m}}}"),
                };
                self.out.push_str(&op);
                if !r.greedy {
                    self.out.push('?');
                }
            }
            Ast::Group(g) => {
                let saved = *f;
                match &g.kind {
                    GroupKind::CaptureIndex(_) => self.out.push('('),
                    GroupKind::CaptureName {
                        starts_with_p,
                        name,
                    } => {
                        let mut cs = name.name.chars();
                        let java_name = cs.next().is_some_and(|c| c.is_ascii_alphabetic())
                            && cs.all(|c| c.is_ascii_alphanumeric());
                        if *starts_with_p || !java_name {
                            return Err(java_rejects(
                                self.text(&g.span),
                                "Java's group names are (?<[a-zA-Z][a-zA-Z0-9]*>..)",
                            ));
                        }
                        self.out.push_str(&format!("(?<{}>", name.name));
                    }
                    GroupKind::NonCapturing(flags) => {
                        self.apply_flags(flags, f)?;
                        self.out.push_str("(?:");
                    }
                }
                self.walk(&g.ast, f)?;
                self.out.push(')');
                *f = saved;
            }
            Ast::Alternation(alt) => {
                self.out.push_str("(?:");
                for (i, x) in alt.asts.iter().enumerate() {
                    if i > 0 {
                        self.out.push('|');
                    }
                    // Java: a flag set in one alternative holds in the next.
                    self.walk(x, f)?;
                }
                self.out.push(')');
            }
            Ast::Concat(c) => {
                let mut i = 0;
                while i < c.asts.len() {
                    // A run of two or more literals is one Java `Slice`.
                    let run = c.asts[i..].iter().take_while(|x| literal_like(x)).count();
                    if run >= 2 {
                        for x in &c.asts[i..i + run] {
                            match x {
                                Ast::Literal(lit) => {
                                    let ch = self.literal_char(lit)?;
                                    self.emit_literal(ch, *f, true);
                                }
                                Ast::Assertion(x) => self.assertion(x, *f)?,
                                _ => unreachable!("literal_like"),
                            }
                        }
                        i += run;
                    } else {
                        self.walk(&c.asts[i], f)?;
                        i += 1;
                    }
                }
            }
        }
        Ok(())
    }

    fn assertion(&mut self, x: &ast::Assertion, f: JFlags) -> Result<(), AnalysisError> {
        let what = self.text(&x.span);
        match x.kind {
            AssertionKind::StartLine | AssertionKind::EndLine if f.multiline => {
                return Err(unsupported(what, "MULTILINE ^ and $"));
            }
            AssertionKind::StartLine | AssertionKind::StartText => self.out.push_str(r"\A"),
            AssertionKind::EndText => self.out.push_str(r"\z"),
            AssertionKind::EndLine => {
                self.has_dollar = true;
                self.out.push_str(if self.sentinel {
                    r"(?:\z|(?-u:\xFF))"
                } else {
                    r"\z"
                });
            }
            // `\<` and `\>` are escaped literals in Java.
            AssertionKind::WordBoundaryStartAngle => self.emit_literal(u32::from('<'), f, false),
            AssertionKind::WordBoundaryEndAngle => self.emit_literal(u32::from('>'), f, false),
            AssertionKind::WordBoundary | AssertionKind::NotWordBoundary => {
                return Err(unsupported(
                    what,
                    "Java's word boundary counts a non-spacing mark after a letter as a word character",
                ))
            }
            // `\b{start}` and the like.
            _ => return Err(java_rejects(what, "Java's \\b takes no braces")),
        }
        Ok(())
    }
}

/// A literal, or `\<`/`\>` (literals in Java): what Java's `atom()` folds
/// into one `Slice`.
fn literal_like(a: &Ast) -> bool {
    match a {
        Ast::Literal(_) => true,
        Ast::Assertion(x) => matches!(
            x.kind,
            AssertionKind::WordBoundaryStartAngle | AssertionKind::WordBoundaryEndAngle
        ),
        _ => false,
    }
}

/// An `&&` operand Java parses differently: empty, or led or ended by `&`.
fn operand_is_quirky(side: &ClassSet) -> bool {
    let amp = |i: &ClassSetItem| matches!(i, ClassSetItem::Literal(l) if l.c == '&');
    match side {
        ClassSet::Item(ClassSetItem::Empty(_)) => true,
        ClassSet::Item(ClassSetItem::Union(u)) => {
            u.items.is_empty()
                || u.items.first().is_some_and(amp)
                || u.items.last().is_some_and(amp)
        }
        ClassSet::Item(i) => amp(i),
        ClassSet::BinaryOp(_) => false,
    }
}

/// Whether `a` can match the empty string.
fn nullable(a: &Ast) -> bool {
    match a {
        Ast::Empty(_) | Ast::Flags(_) | Ast::Assertion(_) => true,
        Ast::Literal(_) | Ast::Dot(_) => false,
        Ast::ClassUnicode(_) | Ast::ClassPerl(_) | Ast::ClassBracketed(_) => false,
        Ast::Repetition(r) => {
            matches!(
                r.op.kind,
                RepetitionKind::ZeroOrOne
                    | RepetitionKind::ZeroOrMore
                    | RepetitionKind::Range(
                        RepetitionRange::Exactly(0)
                            | RepetitionRange::AtLeast(0)
                            | RepetitionRange::Bounded(0, _)
                    )
            ) || nullable(&r.ast)
        }
        Ast::Group(g) => nullable(&g.ast),
        Ast::Alternation(x) => x.asts.iter().any(nullable),
        Ast::Concat(x) => x.asts.iter().all(nullable),
    }
}

/// Whether a repetition makes Java's captures differ from the `regex`
/// crate's: a counted or starred repetition holding a capture inside
/// another repetition (Java's `GroupCurly` records its own span again after
/// the rest of the match succeeds), or holding one below its own group (a
/// first-match iteration can leave it set from a failed attempt).
fn capture_quirk(a: &Ast, in_rep: bool) -> bool {
    match a {
        Ast::Repetition(r) => {
            let counted = !matches!(r.op.kind, RepetitionKind::ZeroOrOne);
            if counted {
                let nested = match &*r.ast {
                    Ast::Group(g) => has_capture(&g.ast),
                    other => has_capture(other),
                };
                if nested || (in_rep && has_capture(&r.ast)) {
                    return true;
                }
            }
            capture_quirk(&r.ast, in_rep || counted)
        }
        Ast::Group(g) => capture_quirk(&g.ast, in_rep),
        Ast::Alternation(x) => x.asts.iter().any(|a| capture_quirk(a, in_rep)),
        Ast::Concat(x) => x.asts.iter().any(|a| capture_quirk(a, in_rep)),
        _ => false,
    }
}

/// Whether `a` holds a capturing group.
fn has_capture(a: &Ast) -> bool {
    match a {
        Ast::Repetition(r) => has_capture(&r.ast),
        Ast::Group(g) => g.is_capturing() || has_capture(&g.ast),
        Ast::Alternation(x) => x.asts.iter().any(has_capture),
        Ast::Concat(x) => x.asts.iter().any(has_capture),
        _ => false,
    }
}

// ----------------------------------------------------------------- patterns

/// What Java does in a search that starts between the surrogates of a pair
/// (only after an empty match before a supplementary character).
#[derive(Debug)]
enum Mid {
    /// The pattern cannot match empty: never reached.
    Never,
    /// No class holds a lone surrogate: an empty match or none, with the
    /// groups that take part in it.
    Empty(Option<Vec<bool>>),
    /// Some class holds the low surrogates: replay with [`LOW_HALF`].
    Search(Regex),
}

#[derive(Debug)]
struct Compiled {
    /// `$` is `\z`: for a text that does not end in a line terminator.
    plain: Regex,
    plain_whole: Regex,
    /// With the sentinel: for a text that does, when the pattern has `$`.
    dollar: Option<(Regex, Regex)>,
    mid: Mid,
    names: HashMap<String, usize>,
    groups: usize,
}

/// Which matcher runs a pattern.
#[derive(Debug, Clone)]
enum Engine {
    /// The `regex` crate (see the module docs).
    Shim(Arc<Compiled>),
    /// The backtracking matcher, for what the shim refuses.
    Backtrack(Arc<Program>),
}

/// A compiled `java.util.regex.Pattern`.
#[derive(Debug, Clone)]
pub struct JavaPattern {
    engine: Engine,
}

/// The `regex` crate's compilation of a translated pattern: what fails
/// here (its size limit) is this shim's limit, not Java's.
fn build(p: &str) -> Result<Regex, AnalysisError> {
    Regex::new(p).map_err(|e| unsupported("pattern", &e.to_string()))
}

fn translate<'s>(
    src: &'s str,
    a: &Ast,
    sentinel: bool,
    mid: bool,
) -> Result<Translator<'s>, AnalysisError> {
    let mut t = Translator {
        src,
        sentinel,
        mid,
        out: String::with_capacity(src.len() * 4),
        has_dollar: false,
        has_low_class: false,
    };
    t.walk(a, &mut JFlags::default())?;
    Ok(t)
}

/// The `$` assertions of `a`, a repeated one counted twice.
fn dollars(a: &Ast, repeated: bool) -> usize {
    match a {
        Ast::Assertion(x) if x.kind == AssertionKind::EndLine => 1 + usize::from(repeated),
        Ast::Repetition(r) => dollars(&r.ast, true),
        Ast::Group(g) => dollars(&g.ast, repeated),
        Ast::Alternation(x) => x.asts.iter().map(|a| dollars(a, repeated)).sum(),
        Ast::Concat(x) => x.asts.iter().map(|a| dollars(a, repeated)).sum(),
        _ => 0,
    }
}

/// The byte index of the line terminator Java's non-multiline `$` may
/// match before, when `text` ends in one.
fn final_terminator(text: &str) -> Option<usize> {
    if text.ends_with("\r\n") {
        return Some(text.len() - 2);
    }
    let last = text.chars().next_back()?;
    matches!(last, '\n' | '\r' | '\u{85}' | '\u{2028}' | '\u{2029}')
        .then(|| text.len() - last.len_utf8())
}

fn with_sentinel(text: &str, at: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(text.len() + 1);
    v.extend_from_slice(&text.as_bytes()[..at]);
    v.push(SENTINEL);
    v.extend_from_slice(&text.as_bytes()[at..]);
    v
}

impl JavaPattern {
    /// `Pattern.compile(String)`.
    pub fn compile(pattern: &str) -> Result<Self, AnalysisError> {
        // The shim first; whatever it refuses, the backtracking matcher
        // compiles -- or rejects with Java's own reason.
        match Self::compile_shim(pattern) {
            Ok(p) => Ok(p),
            Err(_) => Ok(JavaPattern {
                engine: Engine::Backtrack(Arc::new(java_backtrack::compile(pattern)?)),
            }),
        }
    }

    /// Whether the backtracking matcher runs this pattern (the shim refused
    /// it).
    pub fn is_backtracking(&self) -> bool {
        matches!(self.engine, Engine::Backtrack(_))
    }

    fn compile_shim(pattern: &str) -> Result<Self, AnalysisError> {
        let src = prepass(pattern);
        let parsed = ast::parse::Parser::new()
            .parse(&src)
            .map_err(|e| parse_error(&src, e))?;
        if capture_quirk(&parsed, false) {
            return Err(unsupported(
                pattern,
                "a repeated group holding a capture: Java's captures there are not the `regex` crate's",
            ));
        }
        if dollars(&parsed, false) > 1 {
            // The sentinel stands for one zero-width `$`: a second one, or
            // a repeated one, would need it twice.
            return Err(unsupported("$", "more than one `$` at a position"));
        }
        let plain_t = translate(&src, &parsed, false, false)?;
        let plain = build(&plain_t.out)?;
        let plain_whole = build(&format!(r"\A(?:{})\z", plain_t.out))?;
        let dollar = if plain_t.has_dollar {
            let s = translate(&src, &parsed, true, false)?;
            Some((build(&s.out)?, build(&format!(r"\A(?:{})\z", s.out))?))
        } else {
            None
        };
        let hir = regex_syntax::Parser::new()
            .parse(&plain_t.out)
            .map_err(|e| unsupported(pattern, &e.to_string()))?;
        let mid = if hir.properties().minimum_len() != Some(0) {
            Mid::Never
        } else if plain_t.has_low_class {
            Mid::Search(build(
                &translate(&src, &parsed, plain_t.has_dollar, true)?.out,
            )?)
        } else {
            let mut locs = plain.capture_locations();
            let hay = [PAD, LOW_HALF];
            Mid::Empty(
                plain
                    .captures_read_at(&mut locs, &hay, 1)
                    .filter(|m| m.start() == 1)
                    .map(|_| (0..locs.len()).map(|g| locs.get(g).is_some()).collect()),
            )
        };
        let names = plain
            .capture_names()
            .enumerate()
            .filter_map(|(i, n)| n.map(|n| (n.to_string(), i)))
            .collect();
        Ok(JavaPattern {
            engine: Engine::Shim(Arc::new(Compiled {
                groups: plain.captures_len(),
                plain,
                plain_whole,
                dollar,
                mid,
                names,
            })),
        })
    }

    /// The shim's compilation, for a pattern it runs.
    fn shim(&self) -> Option<&Arc<Compiled>> {
        match &self.engine {
            Engine::Shim(c) => Some(c),
            Engine::Backtrack(_) => None,
        }
    }

    /// Capturing groups, group 0 included.
    fn group_total(&self) -> usize {
        match &self.engine {
            Engine::Shim(c) => c.groups,
            Engine::Backtrack(p) => p.group_total(),
        }
    }

    /// A named group's number.
    fn group_named(&self, name: &str) -> Option<usize> {
        match &self.engine {
            Engine::Shim(c) => c.names.get(name).copied(),
            Engine::Backtrack(p) => p.group_named(name),
        }
    }

    /// The backtracking matcher's answer for `s`: `matches()` (`whole`) or
    /// a first `find()`.
    fn backtrack(p: &Program, s: &str, whole: bool) -> Result<bool, AnalysisError> {
        let units: Vec<u16> = s.encode_utf16().collect();
        let mut st = State::new(&units, p.group_total(), 0);
        if whole {
            p.matches_all(&mut st)
        } else {
            p.search(0, &mut st)
        }
    }

    /// `matcher(s).matches()`.
    pub fn matches(&self, s: &str) -> bool {
        self.try_matches(s).unwrap_or(false)
    }

    /// `matcher(s).matches()`; `Err` where Java's backtracking overflows
    /// its stack.
    pub fn try_matches(&self, s: &str) -> Result<bool, AnalysisError> {
        let inner = match &self.engine {
            Engine::Shim(c) => c,
            Engine::Backtrack(p) => return Self::backtrack(p, s, true),
        };
        Ok(match (&inner.dollar, final_terminator(s)) {
            (Some((_, whole)), Some(at)) => whole.is_match(&with_sentinel(s, at)),
            _ => inner.plain_whole.is_match(s.as_bytes()),
        })
    }

    /// `matcher(s).find()` from the start.
    pub fn find_in(&self, s: &str) -> bool {
        let inner = match &self.engine {
            Engine::Shim(c) => c,
            Engine::Backtrack(p) => return Self::backtrack(p, s, false).unwrap_or(false),
        };
        match (&inner.dollar, final_terminator(s)) {
            (Some((re, _)), Some(at)) => re.is_match(&with_sentinel(s, at)),
            _ => inner.plain.is_match(s.as_bytes()),
        }
    }

    /// `Matcher.replaceAll`/`replaceFirst` with Java's replacement syntax.
    pub fn replace(&self, s: &str, replacement: &str, all: bool) -> Result<String, AnalysisError> {
        JavaMatcher::new(self, s).replace(replacement, all)
    }
}

// ------------------------------------------------------------------ matcher

/// A `java.util.regex.Matcher` over one text: `find()` from where the last
/// match ended (one UTF-16 unit on after an empty match), and
/// `start(group)`/`end(group)` in UTF-16 units (`-1` for a group that did
/// not participate).
#[derive(Debug, Clone)]
pub struct JavaMatcher {
    pattern: JavaPattern,
    text: String,
    /// The text with the `$` sentinel before its final line terminator, when
    /// the pattern has `$` and the text ends in one.
    hay: Option<Vec<u8>>,
    sentinel: Option<usize>,
    /// The UTF-16 offset of each byte's character (and of the end), so it
    /// is non-decreasing; empty when the text is ASCII.
    utf16_at: Vec<i32>,
    len16: i32,
    last: Option<(i32, i32)>,
    done: bool,
    groups: Vec<(i32, i32)>,
    /// Reused by every search (no allocation per match).
    locs: CaptureLocations,
    /// The haystack a mid-pair replay writes its marker bytes into.
    scratch: Vec<u8>,
    /// The last search's byte spans (reused).
    spans: Vec<Option<(usize, usize)>>,
    /// The text's UTF-16 units, for the backtracking matcher.
    units: Vec<u16>,
    /// Where the last match ended, for `\G` (`oldLast`; `None`: reset).
    old_last: Option<usize>,
    /// The backtracking matcher's group slots, kept between searches.
    bt_groups: Vec<i32>,
    /// The backtracking matcher's stacks, kept between `find()`s.
    bt_scratch: Scratch,
}

impl JavaMatcher {
    /// `pattern.matcher(text)`.
    pub fn new(pattern: &JavaPattern, text: &str) -> Self {
        let mut m = JavaMatcher {
            pattern: pattern.clone(),
            locs: match &pattern.engine {
                Engine::Shim(c) => c.plain.capture_locations(),
                Engine::Backtrack(_) => EMPTY_LOCS.capture_locations(),
            },
            text: String::new(),
            hay: None,
            sentinel: None,
            utf16_at: Vec::new(),
            len16: 0,
            last: None,
            done: false,
            groups: Vec::new(),
            scratch: Vec::new(),
            spans: Vec::new(),
            units: Vec::new(),
            old_last: None,
            bt_groups: Vec::new(),
            bt_scratch: Scratch::default(),
        };
        m.reset(text);
        m
    }

    /// `reset(CharSequence)`.
    pub fn reset(&mut self, text: &str) {
        self.text.clear();
        self.text.push_str(text);
        self.utf16_at.clear();
        if text.is_ascii() {
            self.len16 = text.len() as i32;
        } else {
            self.utf16_at.reserve(text.len() + 1);
            let mut u = 0i32;
            for c in text.chars() {
                for _ in 0..c.len_utf8() {
                    self.utf16_at.push(u);
                }
                u += c.len_utf16() as i32;
            }
            self.utf16_at.push(u);
            self.len16 = u;
        }
        self.sentinel = None;
        self.hay = None;
        if let Engine::Backtrack(_) = self.pattern.engine {
            self.units.clear();
            self.units.extend(text.encode_utf16());
        }
        if self.pattern.shim().is_some_and(|c| c.dollar.is_some()) {
            if let Some(at) = final_terminator(text) {
                self.sentinel = Some(at);
                self.hay = Some(with_sentinel(text, at));
            }
        }
        self.scratch.clear();
        self.rewind();
    }

    fn rewind(&mut self) {
        self.last = None;
        self.done = false;
        self.groups.clear();
        self.old_last = None;
    }

    /// The text being matched.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The text's length in UTF-16 units.
    pub fn len_utf16(&self) -> i32 {
        self.len16
    }

    /// `groupCount()`.
    pub fn group_count(&self) -> usize {
        self.pattern.group_total() - 1
    }

    #[inline]
    fn utf16(&self, b: usize) -> i32 {
        match self.utf16_at.get(b) {
            Some(&u) => u,
            None => b as i32,
        }
    }

    /// The byte where UTF-16 offset `u` falls, and whether it falls between
    /// the surrogates of the (supplementary) character starting there.
    #[inline]
    fn locate(&self, u: i32) -> (usize, bool) {
        if self.utf16_at.is_empty() {
            return (u as usize, false);
        }
        let b = self.utf16_at.partition_point(|&x| x < u);
        if self.utf16_at.get(b) == Some(&u) {
            (b, false)
        } else {
            (b - 4, true)
        }
    }

    #[inline]
    fn text_to_hay(&self, b: usize) -> usize {
        match self.sentinel {
            Some(s) if b > s => b + 1,
            _ => b,
        }
    }

    #[inline]
    fn hay_to_text(&self, h: usize) -> usize {
        match self.sentinel {
            Some(s) if h > s => h - 1,
            _ => h,
        }
    }

    /// `find()`; `false` also where Java's backtracking would overflow its
    /// stack ([`Self::try_find`] reports that).
    pub fn find(&mut self) -> bool {
        self.try_find().unwrap_or(false)
    }

    /// `find()`; `Err` where Java's backtracking overflows its stack.
    pub fn try_find(&mut self) -> Result<bool, AnalysisError> {
        if matches!(self.pattern.engine, Engine::Backtrack(_)) {
            return self.find_backtrack();
        }
        Ok(self.find_shim())
    }

    /// `find()` by the backtracking matcher.
    fn find_backtrack(&mut self) -> Result<bool, AnalysisError> {
        let Engine::Backtrack(p) = &self.pattern.engine else {
            return Ok(false);
        };
        if self.done {
            return Ok(false);
        }
        let from = match self.last {
            None => 0,
            Some((s, e)) if s == e => e + 1,
            Some((_, e)) => e,
        };
        self.groups.clear();
        if from > self.len16 {
            self.done = true;
            return Ok(false);
        }
        let from = from as usize;
        let old_last = self.old_last.unwrap_or(from);
        let mut st = State::reusing(
            &self.units,
            std::mem::take(&mut self.bt_groups),
            std::mem::take(&mut self.bt_scratch),
            p.group_total(),
            old_last,
        );
        let found = p.search(from, &mut st);
        let (groups, scratch) = st.into_parts();
        self.bt_scratch = scratch;
        let found = match found {
            Ok(f) => f,
            Err(e) => {
                self.done = true;
                self.bt_groups = groups;
                return Err(e);
            }
        };
        if !found {
            self.done = true;
            self.bt_groups = groups;
            return Ok(false);
        }
        self.groups.clear();
        self.groups
            .extend(groups.chunks_exact(2).map(|g| (g[0], g[1])));
        self.bt_groups = groups;
        self.last = Some(self.groups[0]);
        self.old_last = Some(self.groups[0].1 as usize);
        Ok(true)
    }

    /// `find()` by the shim.
    fn find_shim(&mut self) -> bool {
        if self.done {
            return false;
        }
        let from = match self.last {
            None => 0,
            Some((s, e)) if s == e => e + 1,
            Some((_, e)) => e,
        };
        self.groups.clear();
        if from > self.len16 {
            self.done = true;
            return false;
        }
        let (mut b, mid) = self.locate(from);
        if mid {
            if self.find_mid(b) {
                return true;
            }
            b += 4;
        }
        if self.search_at(self.text_to_hay(b)) {
            true
        } else {
            self.done = true;
            false
        }
    }

    fn search_at(&mut self, h: usize) -> bool {
        let Some(inner) = self.pattern.shim() else {
            return false;
        };
        let (re, hay): (&Regex, &[u8]) = match (&self.hay, &inner.dollar) {
            (Some(hay), Some((re, _))) => (re, hay),
            _ => (&inner.plain, self.text.as_bytes()),
        };
        if inner.groups == 1 {
            // The match alone: no capture search, nothing to convert but two
            // offsets.
            let Some(m) = re.find_at(hay, h) else {
                return false;
            };
            let (s, e) = (m.start(), m.end());
            let span = (self.hay_utf16(s), self.hay_utf16(e));
            self.groups.clear();
            self.groups.push(span);
            self.last = Some(span);
            return true;
        }
        // Reused, like `locs`: a search allocates nothing.
        let mut spans = std::mem::take(&mut self.spans);
        spans.clear();
        if re.captures_read_at(&mut self.locs, hay, h).is_none() {
            self.spans = spans;
            return false;
        }
        spans.extend((0..self.locs.len()).map(|g| self.locs.get(g)));
        self.record(&spans, None);
        self.spans = spans;
        true
    }

    /// The UTF-16 offset of haystack position `p`.
    #[inline]
    fn hay_utf16(&self, p: usize) -> i32 {
        self.utf16(self.hay_to_text(p))
    }

    /// Converts haystack spans to UTF-16 groups; `mid` maps the replay's
    /// marker position to the offset between the surrogates.
    fn record(&mut self, spans: &[Option<(usize, usize)>], mid: Option<(usize, i32)>) {
        let conv = |m: &Self, p: usize| -> i32 {
            match mid {
                Some((at, u)) if p == at => u,
                _ => m.utf16(m.hay_to_text(p)),
            }
        };
        let mut groups = std::mem::take(&mut self.groups);
        groups.clear();
        groups.extend(
            spans
                .iter()
                .map(|s| s.map_or((-1, -1), |(a, b)| (conv(self, a), conv(self, b)))),
        );
        self.last = Some(groups[0]);
        self.groups = groups;
    }

    /// Java's attempt between the surrogates of the character at byte `b`.
    #[cold]
    #[inline(never)]
    fn find_mid(&mut self, b: usize) -> bool {
        let u = self.utf16(b) + 1;
        let Some(inner) = self.pattern.shim().cloned() else {
            return false;
        };
        match &inner.mid {
            Mid::Never | Mid::Empty(None) => false,
            Mid::Empty(Some(took_part)) => {
                self.groups = took_part
                    .iter()
                    .map(|&t| if t { (u, u) } else { (-1, -1) })
                    .collect();
                self.last = Some((u, u));
                true
            }
            Mid::Search(re) => {
                if self.scratch.is_empty() {
                    let hay = self.hay.as_deref().unwrap_or(self.text.as_bytes());
                    self.scratch.extend_from_slice(hay);
                }
                let at = self.text_to_hay(b) + 3;
                let saved = (self.scratch[at - 1], self.scratch[at]);
                self.scratch[at - 1] = PAD;
                self.scratch[at] = LOW_HALF;
                let found = re
                    .captures_read_at(&mut self.locs, &self.scratch, at)
                    .is_some_and(|m| m.start() == at);
                let spans: Vec<Option<(usize, usize)>> =
                    (0..self.locs.len()).map(|g| self.locs.get(g)).collect();
                self.scratch[at - 1] = saved.0;
                self.scratch[at] = saved.1;
                if found {
                    self.record(&spans, Some((at, u)));
                }
                found
            }
        }
    }

    /// `start(group)` in UTF-16 units, `-1` if the group did not match.
    #[inline]
    pub fn start(&self, group: usize) -> i32 {
        self.groups.get(group).map_or(-1, |g| g.0)
    }

    /// `end(group)` in UTF-16 units, `-1` if the group did not match.
    #[inline]
    pub fn end(&self, group: usize) -> i32 {
        self.groups.get(group).map_or(-1, |g| g.1)
    }

    /// `group(group)`: `None` if it did not match.
    pub fn group(&self, group: usize) -> Option<Cow<'_, str>> {
        match self.groups.get(group) {
            Some(&(s, e)) if s >= 0 => Some(self.slice(s, e)),
            _ => None,
        }
    }

    /// The text between UTF-16 offsets `a..b`; half of a surrogate pair cut
    /// off by either end is U+FFFD.
    #[inline]
    pub fn slice(&self, a: i32, b: i32) -> Cow<'_, str> {
        if a >= b {
            return Cow::Borrowed("");
        }
        let (ba, a_mid) = self.locate(a);
        let (bb, b_mid) = self.locate(b);
        if !a_mid && !b_mid {
            return Cow::Borrowed(&self.text[ba..bb]);
        }
        let mut out = String::new();
        let mut from = ba;
        if a_mid {
            out.push('\u{FFFD}');
            from = ba + 4;
        }
        if from < bb {
            out.push_str(&self.text[from..bb]);
        }
        if b_mid {
            out.push('\u{FFFD}');
        }
        Cow::Owned(out)
    }

    /// [`Self::slice`] as UTF-16 units, a cut pair keeping its own half.
    pub fn slice_utf16_into(&self, a: i32, b: i32, out: &mut Vec<u16>) {
        if a >= b {
            return;
        }
        let (ba, a_mid) = self.locate(a);
        let (bb, b_mid) = self.locate(b);
        let mut buf = [0u16; 2];
        let mut from = ba;
        if a_mid {
            let c = self.text[ba..].chars().next().expect("a pair starts here");
            out.push(c.encode_utf16(&mut buf)[1]);
            from = ba + 4;
        }
        if from < bb {
            out.extend(self.text[from..bb].encode_utf16());
        }
        if b_mid {
            let c = self.text[bb..].chars().next().expect("a pair starts here");
            out.push(c.encode_utf16(&mut buf)[0]);
        }
    }

    /// `appendReplacement`'s expansion of `replacement` for the last match.
    pub fn expand_replacement(
        &self,
        replacement: &str,
        out: &mut dyn ReplacementSink,
    ) -> Result<(), AnalysisError> {
        let bad = |m: &str| AnalysisError::IllegalArgument(m.to_string());
        let rep: Vec<char> = replacement.chars().collect();
        let groups = self.pattern.group_total();
        let mut i = 0;
        while i < rep.len() {
            let c = rep[i];
            if c == '\\' {
                i += 1;
                let Some(&n) = rep.get(i) else {
                    return Err(bad("character to be escaped is missing"));
                };
                out.push_char(n);
                i += 1;
            } else if c == '$' {
                i += 1;
                let Some(&n) = rep.get(i) else {
                    return Err(bad("Illegal group reference: group index is missing"));
                };
                let group = if n == '{' {
                    let close = rep[i..].iter().position(|&x| x == '}').map(|p| p + i);
                    let Some(close) = close else {
                        return Err(bad("named capturing group is missing trailing '}'"));
                    };
                    let name: String = rep[i + 1..close].iter().collect();
                    i = close + 1;
                    match self.pattern.group_named(&name) {
                        Some(g) => g,
                        None => return Err(bad(&format!("No group with name {{{name}}}"))),
                    }
                } else {
                    let Some(first) = n.to_digit(10) else {
                        return Err(bad("Illegal group reference"));
                    };
                    let mut group = first as usize;
                    i += 1;
                    // Java: take further digits while the group number stays valid.
                    while let Some(d) = rep.get(i).and_then(|c| c.to_digit(10)) {
                        let next = group * 10 + d as usize;
                        if next >= groups {
                            break;
                        }
                        group = next;
                        i += 1;
                    }
                    if group >= groups {
                        return Err(bad(&format!("No group {group}")));
                    }
                    group
                };
                if let Some(&(s, e)) = self.groups.get(group).filter(|g| g.0 >= 0) {
                    out.push_slice(self, s, e);
                }
            } else {
                out.push_char(c);
                i += 1;
            }
        }
        Ok(())
    }

    /// `replaceAll`/`replaceFirst`: rewinds, then replaces.
    pub fn replace(&mut self, replacement: &str, all: bool) -> Result<String, AnalysisError> {
        self.rewind();
        let mut out = String::with_capacity(self.text.len());
        let mut last = 0;
        while self.find() {
            let (s, e) = (self.start(0), self.end(0));
            out.push_str(&self.slice(last, s));
            self.expand_replacement(replacement, &mut out)?;
            last = e;
            if !all {
                break;
            }
        }
        out.push_str(&self.slice(last, self.len16));
        Ok(out)
    }
}

/// The capture locations of a matcher the shim does not run.
static EMPTY_LOCS: LazyLock<Regex> = LazyLock::new(|| Regex::new("").expect("an empty pattern"));

/// Where [`JavaMatcher::expand_replacement`] writes: a `String` (a cut
/// surrogate half is U+FFFD) or UTF-16 units (it is the unit itself).
pub trait ReplacementSink {
    /// Appends one character.
    fn push_char(&mut self, c: char);
    /// Appends the matcher's text between UTF-16 offsets `a..b`.
    fn push_slice(&mut self, m: &JavaMatcher, a: i32, b: i32);
}

impl ReplacementSink for String {
    fn push_char(&mut self, c: char) {
        self.push(c);
    }

    fn push_slice(&mut self, m: &JavaMatcher, a: i32, b: i32) {
        self.push_str(&m.slice(a, b));
    }
}

impl ReplacementSink for Vec<u16> {
    fn push_char(&mut self, c: char) {
        let mut buf = [0u16; 2];
        self.extend_from_slice(c.encode_utf16(&mut buf));
    }

    fn push_slice(&mut self, m: &JavaMatcher, a: i32, b: i32) {
        m.slice_utf16_into(a, b, self);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finds(p: &str, s: &str) -> Vec<(i32, i32)> {
        let p = JavaPattern::compile(p).unwrap();
        let mut m = JavaMatcher::new(&p, s);
        let mut out = Vec::new();
        while m.find() {
            out.push((m.start(0), m.end(0)));
        }
        out
    }

    fn rejected(p: &str) -> String {
        match JavaPattern::compile(p) {
            Err(AnalysisError::IllegalArgument(m)) => m,
            other => panic!("{p} compiled: {other:?}"),
        }
    }

    #[test]
    fn java_classes_are_ascii() {
        let p = JavaPattern::compile("\\w+").unwrap();
        assert!(p.matches("abc_9"));
        assert!(!p.matches("é"));
        let p = JavaPattern::compile("[\\d\\s]+").unwrap();
        assert!(p.matches("1 2\t3\u{b}"));
        assert!(!p.matches("١"));
        let p = JavaPattern::compile("\\D\\W\\S").unwrap();
        assert!(p.matches("a-x"));
        let p = JavaPattern::compile("[\\W]").unwrap();
        assert!(p.matches("é") && !p.matches("a"));
        let p = JavaPattern::compile("[^\\W]").unwrap();
        assert!(p.matches("a") && !p.matches("é"));
        let p = JavaPattern::compile("[]a]+").unwrap();
        assert!(p.matches("]a]"));
        let p = JavaPattern::compile("[^]a]").unwrap();
        assert!(p.matches("b") && !p.matches("]"));
        assert!(JavaPattern::compile("\\.").unwrap().matches("."));
        assert!(JavaPattern::compile("(a").is_err());
        assert!(JavaPattern::compile("a\\").is_err());
        let never = JavaPattern::compile("[^\\x{0}-\\x{10FFFF}]").unwrap();
        assert!(!never.find_in("a\u{10FFFF}") && !never.matches(""));
        for (p, yes, no) in [
            ("\\p{Alpha}", "a", "é"),
            ("\\p{Upper}", "A", "É"),
            ("\\p{Lower}", "a", "é"),
            ("\\p{Punct}", "!", "¡"),
            ("\\p{Space}", "\u{b}", "\u{a0}"),
            ("\\p{Digit}", "1", "١"),
            ("\\p{Alnum}", "1", "é"),
            ("\\p{Graph}", "~", " "),
            ("\\p{Print}", " ", "\u{7f}"),
            ("\\p{Blank}", "\t", "\n"),
            ("\\p{Cntrl}", "\u{7f}", "a"),
            ("\\p{XDigit}", "f", "g"),
            ("\\p{ASCII}", "\u{7f}", "é"),
            ("\\p{L}", "é", "1"),
            ("\\pL", "é", "1"),
            ("\\p{IsL}", "é", "1"),
            ("\\p{IsLu}", "É", "é"),
            ("\\p{gc=Nd}", "١", "a"),
            ("\\p{general_category=Lu}", "A", "a"),
            ("\\p{LC}", "ǅ", "ʰ"),
            ("\\p{IsLC}", "ǅ", "ʰ"),
            ("\\p{LD}", "١", "_"),
            ("\\p{L1}", "ÿ", "Ā"),
            ("\\p{all}", "\u{10FFFF}", ""),
            ("\\P{L}", "1", "a"),
            ("[\\p{L}&&\\p{Lu}]", "A", "a"),
            ("[\\p{L}&&[^\\p{Lu}]]", "a", "A"),
            ("[a-z&&[^aeiou]]", "b", "a"),
            ("[^a-c&&b-d]", "a", "b"),
            ("[[:alpha:]]", ":", "b"),
            ("[[:^alpha:]]", "^", "b"),
            ("[x[:alpha:]]", "x", "b"),
            ("\\h", "\u{3000}", "\n"),
            ("\\H", "a", "\t"),
            ("\\v", "\u{2028}", " "),
            ("\\V", " ", "\u{b}"),
            ("[\\v]", "\u{85}", "a"),
            ("\\e", "\u{1b}", "e"),
            ("\\Qa.b\\E", "a.b", "axb"),
            ("[\\Q-]\\E]", "]", "a"),
            ("\\Qa", "a", "b"),
            ("\\uD83D\\uDE00", "😀", "a"),
            ("\\x{1F600}", "😀", "a"),
            ("\\u00e9", "é", "e"),
            ("\\<a\\>", "<a>", "a"),
            (".", "a", "\u{85}"),
            ("(?s).", "\u{2028}", ""),
            ("(?<ab1>x)", "x", "y"),
            ("a{2}", "aa", "a"),
            ("a{1,}", "aa", ""),
            ("a{0,1}?", "a", "aa"),
        ] {
            let c = JavaPattern::compile(p).unwrap();
            assert!(c.matches(yes), "{p} should match {yes:?}");
            assert!(!c.matches(no), "{p} should not match {no:?}");
        }
    }

    #[test]
    fn case_insensitivity_is_javas() {
        for (p, s, want) in [
            ("(?i)a", "aA", 2),
            ("(?i)é", "éÉ", 1),
            ("(?iu)é", "éÉ", 2),
            ("(?i)[a-c]", "aBcD", 3),
            ("(?i)[é]", "éÉ", 1),
            ("(?iu)[é]", "éÉ", 2),
            ("(?i)k", "kK\u{212A}", 2),
            ("(?iu)k", "kK\u{212A}", 3),
            ("(?iu)\u{212A}", "kK", 2),
            ("(?i)[a-z]", "\u{212A}\u{17F}", 0),
            ("(?iu)[a-z]", "\u{212A}\u{17F}", 2),
            ("(?iu)[^k]", "\u{212A}", 0),
            ("(?i)[^a]", "aAb", 1),
            ("(?iu)[ÿ]", "Ÿ", 1),
            ("(?iu)[µ]", "μΜ", 2),
            ("(?iu)i", "\u{130}\u{131}I", 3),
            ("(?iu)[\u{130}]", "iI\u{131}", 3),
            ("(?iu)ß", "\u{1E9E}", 0),
            ("(?iu)ßx", "\u{1E9E}x", 1),
            ("(?iu)\u{1E9E}", "ß", 1),
            ("(?iu)[ß]", "\u{1E9E}", 0),
            ("(?iu)[à-å]", "Å\u{212B}", 2),
            ("(?i)[à-å]", "Å", 0),
            ("(?iu)[À]", "à", 1),
            ("(?i)[À]", "à", 0),
            ("(?iu)ǅ", "ǄǆǅX", 3),
            ("(?iu)[Ǆ-Ǆ]", "ǅǆ", 2),
            ("(?i)\\p{Lu}", "aAǅʰ", 3),
            ("(?i)\\P{Lu}", "aA1", 1),
            ("(?i)\\p{Upper}", "aAé", 2),
            ("(?iu)\\w", "\u{212A}", 0),
            ("(?i)[a&&b]", "A", 0),
            ("(?i)[a-c&&B]", "b", 1),
            ("(?i:a)b", "AbAB", 1),
            ("a(?i)b|c", "aBC", 2),
            ("(?i)(?-i:a)", "A", 0),
            ("(?i)(?u)k", "\u{212A}", 1),
            ("(?i)\\<A", "<a", 1),
        ] {
            assert_eq!(finds(p, s).len(), want, "{p} on {s:?}");
        }
    }

    #[test]
    fn dollar_matches_before_a_final_line_terminator() {
        assert_eq!(finds("$", "a\n"), vec![(1, 1), (2, 2)]);
        assert_eq!(finds("$", "a\r\n"), vec![(1, 1), (3, 3)]);
        assert_eq!(finds("$", "a\r"), vec![(1, 1), (2, 2)]);
        assert_eq!(finds("$", "a\u{85}"), vec![(1, 1), (2, 2)]);
        assert_eq!(finds("$", "a\u{2028}"), vec![(1, 1), (2, 2)]);
        assert_eq!(finds("$", "a\n\n"), vec![(2, 2), (3, 3)]);
        assert_eq!(finds("$", "a"), vec![(1, 1)]);
        assert_eq!(finds("$", ""), vec![(0, 0)]);
        assert_eq!(finds("a$\\n", "a\n"), vec![(0, 2)]);
        assert_eq!(finds("\\s+", "a \n"), vec![(1, 3)]);
        assert_eq!(finds("\\n", "a\n"), vec![(1, 2)]);
        assert_eq!(finds("x*", "é\n"), vec![(0, 0), (1, 1), (2, 2)]);
        let p = JavaPattern::compile("^a$").unwrap();
        assert!(p.matches("a") && !p.matches("a\n") && p.find_in("a\n"));
        assert!(!p.find_in("a\nb"));
        let p = JavaPattern::compile("a$\\n").unwrap();
        assert!(p.matches("a\n"));
        assert_eq!(p.replace("a\r\n", "<$0>", true).unwrap(), "a\r\n");
        assert_eq!(
            JavaPattern::compile("(a)$")
                .unwrap()
                .replace("ba\n", "[$1]", true)
                .unwrap(),
            "b[a]\n"
        );
    }

    #[test]
    fn rejects_what_differs_from_java() {
        // Java compiles these (`Pattern.compile`, JDK 21); the shim cannot.
        for p in [
            "\\bfox",
            "\\B",
            "(?m)^a",
            "(?m)a$",
            "(?x)a",
            "(?U)a",
            "[a~~b]",
            "[&&a]",
            "[a&&]",
            "[a&&&b]",
            "\\p{IsLatin}",
            "\\p{InGreek}",
            "\\p{IsAlphabetic}",
            "\\p{javaLowerCase}",
            "\\p{sc=Latin}",
            "(a|)*",
            "(a*)+",
            "(a?){2}",
            "a++",
            "a*+",
            "a**",
            "(?<=a)b",
            "(a)\\1",
            "\\Z",
            "\\G",
            "\\R",
            "\\X",
            "\\cA",
            "\\0101",
            "\\uD800",
            "\\N{LATIN SMALL LETTER A}",
            "(?>ab)",
        ] {
            let e = JavaPattern::compile_shim(p).unwrap_err();
            assert!(is_unsupported(&e), "{p}: {e}");
            // The backtracking matcher takes them over -- but for two it
            // refuses too, and one Java rejects.
            match (p, JavaPattern::compile(p)) {
                ("\\X" | "\\N{LATIN SMALL LETTER A}", Err(e)) => assert!(is_unsupported(&e), "{p}"),
                ("a**", Err(e)) => {
                    assert!(e.to_string().contains("Dangling meta character"), "{e}")
                }
                (_, Ok(q)) => assert!(q.is_backtracking(), "{p}"),
                (_, Err(e)) => panic!("{p}: {e}"),
            }
        }
        // Java refuses these too; the `regex` crate alone would not.
        for p in [
            "\\b{start}",
            "(?R)a",
            "(?P<n>a)",
            "(?<a_b>a)",
            "\\u{e9}",
            "\\U000000E9",
            "a{ 2 }",
            "a{2, 3}",
            "[a--b]",
            "x{,3}",
        ] {
            let e = JavaPattern::compile(p).unwrap_err();
            assert!(!is_unsupported(&e), "{p}: {e}");
            assert!(rejected(p).contains("PatternSyntaxException"), "{p}");
        }
        // Java refuses these as well; the shim reports its own limit.
        for p in [
            "\\p{Latin}",
            "\\p{gc:Lu}",
            "\\p{Uppercase_Letter}",
            "\\p{ Lu }",
        ] {
            assert!(
                is_unsupported(&JavaPattern::compile_shim(p).unwrap_err()),
                "{p}"
            );
            assert!(
                rejected(p).contains("Unknown character property name"),
                "{p}"
            );
        }
        // A non-repeating or non-capturing nullable body is fine.
        for p in ["(a|)?", "(?:a|)*", "(a*)", "((a)|b)*"] {
            assert!(JavaPattern::compile(p).is_ok(), "{p}");
        }
    }

    /// Java's `find()`: after an empty match it moves one UTF-16 unit on --
    /// which lands between the surrogates of a supplementary character.
    #[test]
    fn matcher_finds_like_java() {
        assert_eq!(
            finds("a*", "baa😀"),
            vec![(0, 0), (1, 3), (3, 3), (4, 4), (5, 5)]
        );
        assert_eq!(
            finds("x*", "abxd"),
            vec![(0, 0), (1, 1), (2, 3), (3, 3), (4, 4)]
        );
        let p = JavaPattern::compile("x*").unwrap();
        assert_eq!(p.replace("abxd", "-", true).unwrap(), "-a-b--d-");
        assert_eq!(p.replace("😀", "-", true).unwrap(), "-\u{FFFD}-\u{FFFD}-");
        assert_eq!(p.replace("abxd", "-", false).unwrap(), "-abxd");
        // A class holding the low surrogates matches the lone half there.
        assert_eq!(finds("\\A|.", "😀"), vec![(0, 0), (1, 2)]);
        let p = JavaPattern::compile("\\A|(.)").unwrap();
        let mut m = JavaMatcher::new(&p, "😀b");
        assert!(m.find());
        assert_eq!((m.start(0), m.end(0), m.start(1)), (0, 0, -1));
        assert!(m.find());
        assert_eq!((m.start(1), m.end(1)), (1, 2));
        assert_eq!(m.group(1).unwrap(), "\u{FFFD}");
        let mut units = Vec::new();
        m.slice_utf16_into(0, 2, &mut units);
        assert_eq!(units, vec![0xD83D, 0xDE00]);
        units.clear();
        m.slice_utf16_into(1, 3, &mut units);
        assert_eq!(units, vec![0xDE00, u16::from(b'b')]);
        units.clear();
        m.slice_utf16_into(0, 1, &mut units);
        assert_eq!(units, vec![0xD83D]);
        assert_eq!(m.slice(0, 1), "\u{FFFD}");
        assert_eq!(m.slice(0, 3), "😀b");
        assert_eq!(m.slice(2, 1), "");
        assert!(m.find());
        assert_eq!(m.group(0).unwrap(), "b");
        assert!(!m.find());
        // Groups that take part in the empty match between the surrogates.
        let p = JavaPattern::compile("(a*)(b)?").unwrap();
        let mut m = JavaMatcher::new(&p, "😀");
        assert!(m.find() && m.find());
        assert_eq!((m.start(0), m.start(1), m.start(2)), (1, 1, -1));
        assert_eq!(m.group(2), None);
        // A pattern that cannot match empty there skips the half.
        assert_eq!(finds("x*$", "😀"), vec![(2, 2)]);
        let p = JavaPattern::compile("(x)|(y)").unwrap();
        let mut m = JavaMatcher::new(&p, "😀y");
        assert_eq!(m.group_count(), 2);
        assert!(m.find());
        assert_eq!((m.start(1), m.end(1), m.start(2), m.end(2)), (-1, -1, 2, 3));
        assert_eq!(m.text(), "😀y");
        assert_eq!(m.len_utf16(), 3);
        assert!(!m.find());
        assert_eq!(m.start(0), -1, "a failed find clears the groups");
        assert!(!m.find(), "an exhausted matcher stays exhausted");
        // The sentinel and the mid-pair replay together.
        assert_eq!(finds("[^a]*$", "😀\n"), vec![(0, 3), (3, 3)]);
        assert_eq!(finds("\\A|[^a]$", "😀\n"), vec![(0, 0), (1, 2), (2, 3)]);
    }

    #[test]
    fn replacement_syntax_is_javas() {
        let p = JavaPattern::compile("([a-z]+)-([a-z]+)").unwrap();
        assert_eq!(
            p.replace("ab-cd ef-gh", "$2_$1", true).unwrap(),
            "cd_ab gh_ef"
        );
        assert_eq!(
            p.replace("ab-cd ef-gh", "$2_$1", false).unwrap(),
            "cd_ab ef-gh"
        );
        assert_eq!(p.replace("ab-cd", "\\$1$10", true).unwrap(), "$1ab0");
        let n = JavaPattern::compile("(?<w>x)").unwrap();
        assert_eq!(n.replace("axb", "[${w}]", true).unwrap(), "a[x]b");
        assert!(n.replace("axb", "${v}", true).is_err());
        assert!(p.replace("ab-cd", "$9", true).is_err());
        assert!(p.replace("ab-cd", "$", true).is_err());
        assert!(p.replace("ab-cd", "$x", true).is_err());
        assert!(p.replace("ab-cd", "x\\", true).is_err());
        assert!(n.replace("x", "${w", true).is_err());
        assert_eq!(p.replace("none", "$1", true).unwrap(), "none");
        let o = JavaPattern::compile("(a)|b").unwrap();
        assert_eq!(o.replace("ab", "[$1]", true).unwrap(), "[a][]");
        let mut units: Vec<u16> = Vec::new();
        let mut m = JavaMatcher::new(&o, "a");
        assert!(m.find());
        m.expand_replacement("<$1é>", &mut units).unwrap();
        assert_eq!(String::from_utf16(&units).unwrap(), "<aé>");
    }

    #[test]
    fn case_index_covers_every_simple_mapping() {
        for c in CASED_LIMIT..=MAX_CP {
            assert_eq!((to_upper_case(c), to_lower_case(c)), (c, c), "{c:#x}");
        }
        assert!(CASE_INDEX.cased.len() > 2000);
    }

    #[test]
    fn code_point_sets() {
        let a = CpSet::from_ranges(vec![(5, 9), (1, 2), (3, 3), (20, 30)]);
        assert_eq!(a.0, vec![(1, 3), (5, 9), (20, 30)]);
        assert_eq!(
            a.intersect(&CpSet::range(2, 25)).0,
            vec![(2, 3), (5, 9), (20, 25)]
        );
        assert_eq!(a.complement().0[0], (0, 0));
        assert_eq!(CpSet::all().complement(), CpSet::default());
        assert_eq!(CpSet::default().complement(), CpSet::all());
        assert!(a.contains(7) && !a.contains(4) && !a.contains(31));
        assert_eq!(prepass("abc"), "abc");
        assert_eq!(prepass("\\uD83D"), "\\uD83D");
        assert_eq!(prepass("a\\"), "a\\");
    }
}
