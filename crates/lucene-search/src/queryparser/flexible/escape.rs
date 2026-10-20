//! `standard.parser.EscapeQuerySyntaxImpl.escape(text, locale, type)`, the
//! `EscapeQuerySyntax` the parser's messages print a node's term with
//! (`toQueryString(new EscapeQuerySyntaxImpl())`).
//!
//! The text's `\` and its escaped `*` and `?` get a backslash
//! (`UnescapedCharSequence.toStringEscaped`); then, in a quoted term,
//! each `"`; in a bare one each of `"<>=!()^[{:]}~/`, a leading `+`, `-` or
//! `@`, each white space character Java lists (` \t\n\r\f\b\u{3000}`), and
//! the whole of a term spelling an operator (`AND`, `OR`, `NOT`, `TO`,
//! `WITHIN`, `SENTENCE`, `PARAGRAPH`, `INORDER`, any case).
//!
//! Each character is found by `escapeIgnoringCase`, which searches the
//! text's `toLowerCase(locale)` and copies from the text itself at the index
//! found there: where lower-casing changes the length before a match (`İ`
//! is two units in most locales, `I` before a dot above one fewer in
//! Turkish, `Ì` three in Lithuanian) the backslash lands elsewhere, or an
//! index past the text's end throws `IndexOutOfBoundsException` -- as
//! here. The locale matters only through its case mapping: the root
//! mapping, Turkish and Azerbaijani's, Lithuanian's
//! ([`CaseLocale`]).
//!
//! Differs: the tailorings' context rules ask a character's canonical
//! combining class, which this crate has no table of: U+0300..U+036F's are
//! written out (230 "above" and the rest), and every other character counts
//! as class 0 -- a combining mark elsewhere (U+20D0, U+1DC0, ...) that Java
//! would look through or treat as "above" ends the context here.

use super::nodes::Text;
use crate::queryparser::classic::QueryParserError;

/// The case mapping a `Locale` gives `String.toLowerCase(locale)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaseLocale {
    /// Every locale without a tailoring (`Locale.ROOT`, `en`, ...).
    #[default]
    Root,
    /// `tr` and `az`.
    Turkic,
    /// `lt`.
    Lithuanian,
}

impl CaseLocale {
    /// The case mapping of a language tag's language.
    pub fn of_tag(tag: &str) -> CaseLocale {
        let language = tag.split(['-', '_']).next().unwrap_or("");
        match language.to_ascii_lowercase().as_str() {
            "tr" | "az" => CaseLocale::Turkic,
            "lt" => CaseLocale::Lithuanian,
            _ => CaseLocale::Root,
        }
    }
}

/// `EscapeQuerySyntax.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EscapeType {
    /// `STRING`: a quoted term.
    String,
    /// `NORMAL`: a bare term.
    Normal,
}

type Result<T> = std::result::Result<T, QueryParserError>;

const BACKSLASH: u16 = 0x5c;

/// `escape(text, locale, type)`.
pub(crate) fn escape(text: &Text, locale: CaseLocale, ty: EscapeType) -> Result<Vec<u16>> {
    if text.is_empty() {
        return Ok(Vec::new());
    }
    // `toStringEscaped(wildcardChars)`
    let mut s = Vec::with_capacity(text.len() + 1);
    for (i, &c) in text.units.iter().enumerate() {
        if c == BACKSLASH || ((c == u16::from(b'*') || c == u16::from(b'?')) && text.was_escaped(i))
        {
            s.push(BACKSLASH);
        }
        s.push(c);
    }
    if ty == EscapeType::String {
        return escape_ignoring_case(s, u16::from(b'"'), locale);
    }
    // `escapeChar`
    for &c in b"\"<>=!()^[{:]}~/" {
        s = escape_ignoring_case(s, u16::from(c), locale)?;
    }
    if matches!(s.first(), Some(&c) if c == u16::from(b'+') || c == u16::from(b'-') || c == u16::from(b'@'))
    {
        s.insert(0, BACKSLASH);
    }
    // `escapeWhiteChar`
    for c in [0x20, 0x09, 0x0a, 0x0d, 0x0c, 0x08, 0x3000] {
        s = escape_ignoring_case(s, c, locale)?;
    }
    const WORDS: [&str; 8] = [
        "AND",
        "OR",
        "NOT",
        "TO",
        "WITHIN",
        "SENTENCE",
        "PARAGRAPH",
        "INORDER",
    ];
    if WORDS.iter().any(|w| equals_ignore_case(&s, w)) {
        s.insert(0, BACKSLASH);
    }
    Ok(s)
}

/// `String.equalsIgnoreCase` against an ASCII word: a unit matches when
/// equal after `toUpperCase`, or after `toLowerCase` of that.
fn equals_ignore_case(s: &[u16], word: &str) -> bool {
    s.len() == word.len()
        && s.iter().zip(word.bytes()).all(|(&c, w)| {
            let up = char::from_u32(u32::from(c)).map(java_upper_char);
            match up {
                Some(u) if u == char::from(w) => true,
                Some(u) => java_lower_char(u) == char::from(w).to_ascii_lowercase(),
                None => false,
            }
        })
}

/// `Character.toUpperCase(char)` for a BMP character: its single-character
/// upper case, or itself.
fn java_upper_char(c: char) -> char {
    let mut up = c.to_uppercase();
    match (up.next(), up.next()) {
        (Some(u), None) => u,
        _ => c,
    }
}

/// `Character.toLowerCase(char)`, likewise -- but `İ`, the one character
/// whose full lower case is longer, whose simple mapping is `i`.
fn java_lower_char(c: char) -> char {
    if c == '\u{130}' {
        return 'i';
    }
    let mut low = c.to_lowercase();
    match (low.next(), low.next()) {
        (Some(l), None) => l,
        _ => c,
    }
}

/// `escapeIgnoringCase(string, c, "\\", locale)` for a one-character
/// sequence (every one `escape` passes).
fn escape_ignoring_case(string: Vec<u16>, c: u16, locale: CaseLocale) -> Result<Vec<u16>> {
    let count = string.len();
    let lowercase = to_lower_case(&string, locale);
    let mut result = Vec::new();
    let mut start = 0;
    let mut copy_start = 0;
    let mut matched = false;
    while start < count {
        let Some(first) = lowercase
            .get(start..)
            .and_then(|r| r.iter().position(|&u| u == c))
            .map(|p| p + start)
        else {
            break;
        };
        append(&mut result, &string, copy_start, first)?;
        result.push(BACKSLASH);
        append(&mut result, &string, first, first + 1)?;
        matched = true;
        copy_start = first + 1;
        start = first + 1;
    }
    if !matched {
        return Ok(string);
    }
    append(&mut result, &string, copy_start, count)?;
    Ok(result)
}

/// `StringBuilder.append(s, start, end)`: `IndexOutOfBoundsException` for a
/// range outside `s`.
fn append(out: &mut Vec<u16>, s: &[u16], start: usize, end: usize) -> Result<()> {
    match s.get(start..end) {
        Some(r) => {
            out.extend_from_slice(r);
            Ok(())
        }
        None => Err(QueryParserError::runtime_class(
            "java.lang.IndexOutOfBoundsException",
            format!(
                "Range [{start}, {end}) out of bounds for length {}",
                s.len()
            ),
        )),
    }
}

/// `String.toLowerCase(locale)`.
pub(crate) fn to_lower_case(s: &[u16], locale: CaseLocale) -> Vec<u16> {
    let root = lucene_analysis::lang::java_string_to_lower_case(s);
    if locale == CaseLocale::Root {
        return root;
    }
    let cps: Vec<u32> = char::decode_utf16(s.iter().copied())
        .map(|r| r.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
        .collect();
    // The root mapping keeps each character's length but `İ`'s (two units):
    // walk it in step to take every character the tailoring leaves alone.
    let mut out = Vec::with_capacity(root.len() + 2);
    let mut at = 0;
    for (i, &cp) in cps.iter().enumerate() {
        let len = if cp == 0x130 || cp > 0xffff { 2 } else { 1 };
        let rooted = root.get(at..at + len).unwrap_or(&[]);
        at += len;
        match (locale, cp) {
            (CaseLocale::Turkic, 0x130) => out.push(0x69),
            (CaseLocale::Turkic, 0x307) if after_i(&cps[..i]) => {}
            (CaseLocale::Turkic, 0x49) if !before_dot(&cps[i + 1..]) => out.push(0x131),
            (CaseLocale::Lithuanian, 0xcc) => out.extend_from_slice(&[0x69, 0x307, 0x300]),
            (CaseLocale::Lithuanian, 0xcd) => out.extend_from_slice(&[0x69, 0x307, 0x301]),
            (CaseLocale::Lithuanian, 0x128) => out.extend_from_slice(&[0x69, 0x307, 0x303]),
            (CaseLocale::Lithuanian, 0x49 | 0x4a | 0x12e) if more_above(&cps[i + 1..]) => {
                out.extend_from_slice(rooted);
                out.push(0x307);
            }
            _ => out.extend_from_slice(rooted),
        }
    }
    out
}

/// The canonical combining class, as far as this module knows it (see the
/// module docs): 230 for the "above" marks of U+0300..U+036F, 1 standing
/// for the block's other non-zero classes, 0 for everything else.
fn combining_class(cp: u32) -> u8 {
    match cp {
        0x300..=0x314
        | 0x33d..=0x344
        | 0x346
        | 0x34a..=0x34c
        | 0x350..=0x352
        | 0x357
        | 0x35b
        | 0x363..=0x36f => 230,
        0x34f => 0,
        0x315..=0x36f => 1,
        _ => 0,
    }
}

/// `AFTER_I`: the last preceding base character is `I`, with no "above"
/// mark between.
fn after_i(before: &[u32]) -> bool {
    for &cp in before.iter().rev() {
        if cp == 0x49 {
            return true;
        }
        if matches!(combining_class(cp), 0 | 230) {
            return false;
        }
    }
    false
}

/// `BEFORE_DOT`: a U+0307 follows, with no base or "above" mark between.
fn before_dot(after: &[u32]) -> bool {
    for &cp in after {
        if cp == 0x307 {
            return true;
        }
        if matches!(combining_class(cp), 0 | 230) {
            return false;
        }
    }
    false
}

/// `MORE_ABOVE`: an "above" mark follows before any base character.
fn more_above(after: &[u32]) -> bool {
    for &cp in after {
        match combining_class(cp) {
            0 => return false,
            230 => return true,
            _ => {}
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn esc(s: &str, locale: CaseLocale, ty: EscapeType) -> String {
        match escape(&Text::plain(s), locale, ty) {
            Ok(u) => String::from_utf16_lossy(&u),
            Err(e) => format!("{}: {}", e.class, e.message),
        }
    }

    #[test]
    fn lower_cases_as_the_jdk() {
        // `toLowerCase(Locale.forLanguageTag(tag))`, JDK 25
        let cases: &[(CaseLocale, &[u16], &[u16])] = &[
            (CaseLocale::Turkic, &[73, 775, 58], &[105, 58]),
            (CaseLocale::Turkic, &[73, 790, 775, 58], &[105, 790, 58]),
            (
                CaseLocale::Turkic,
                &[73, 768, 775, 58],
                &[305, 768, 775, 58],
            ),
            (CaseLocale::Turkic, &[73, 837, 775, 58], &[105, 837, 58]),
            (CaseLocale::Turkic, &[73, 97, 775, 58], &[305, 97, 775, 58]),
            (CaseLocale::Turkic, &[304, 58], &[105, 58]),
            (CaseLocale::Turkic, &[73, 58], &[305, 58]),
            (
                CaseLocale::Turkic,
                &[931, 97, 931, 32, 58],
                &[963, 97, 962, 32, 58],
            ),
            (CaseLocale::Lithuanian, &[204, 58], &[105, 775, 768, 58]),
            (CaseLocale::Lithuanian, &[205], &[105, 775, 769]),
            (CaseLocale::Lithuanian, &[296], &[105, 775, 771]),
            (CaseLocale::Lithuanian, &[73, 768, 58], &[105, 775, 768, 58]),
            (
                CaseLocale::Lithuanian,
                &[73, 790, 768, 58],
                &[105, 775, 790, 768, 58],
            ),
            (
                CaseLocale::Lithuanian,
                &[73, 97, 768, 58],
                &[105, 97, 768, 58],
            ),
            (CaseLocale::Lithuanian, &[74, 769], &[106, 775, 769]),
            (CaseLocale::Lithuanian, &[302, 769], &[303, 775, 769]),
            (CaseLocale::Lithuanian, &[304, 58], &[105, 775, 58]),
            (CaseLocale::Lithuanian, &[73, 789, 58], &[105, 789, 58]),
            (
                CaseLocale::Lithuanian,
                &[73, 0xd83d, 0xde00, 73],
                &[105, 0xd83d, 0xde00, 105],
            ),
            (CaseLocale::Root, &[304, 58], &[105, 775, 58]),
        ];
        for (locale, from, want) in cases {
            assert_eq!(to_lower_case(from, *locale), *want, "{locale:?} {from:?}");
        }
        // Differs: U+20D0 is class 230 in Java (`[105, 775, 8400, 58]`)
        assert_eq!(
            to_lower_case(&[73, 8400, 58], CaseLocale::Lithuanian),
            [105, 8400, 58]
        );
    }

    #[test]
    fn escapes_as_java() {
        use CaseLocale::*;
        use EscapeType::*;
        assert_eq!(esc("", Root, Normal), "");
        assert_eq!(esc("a:b (c)", Root, Normal), "a\\:b\\ \\(c\\)");
        assert_eq!(esc("a\"b\\", Root, String), "a\\\"b\\\\");
        assert_eq!(esc("a:b", Root, String), "a:b");
        assert_eq!(esc("+a", Root, Normal), "\\+a");
        assert_eq!(esc("-a", Root, Normal), "\\-a");
        assert_eq!(esc("@a", Root, Normal), "\\@a");
        assert_eq!(esc("a+", Root, Normal), "a+");
        assert_eq!(esc("and", Root, Normal), "\\and");
        assert_eq!(esc("Within", Root, Normal), "\\Within");
        assert_eq!(esc("ANDY", Root, Normal), "ANDY");
        // `equalsIgnoreCase`'s simple mappings: `İ` lowers to `i`, `ı` uppers to `I`
        assert_eq!(esc("\u{130}NORDER", Root, Normal), "\\\u{130}NORDER");
        assert_eq!(
            esc("w\u{131}th\u{130}n", Root, Normal),
            "\\w\u{131}th\u{130}n"
        );
        assert_eq!(esc("\u{df}", Root, Normal), "\u{df}");
        assert_eq!(esc("a\u{3000}b\tc", Root, Normal), "a\\\u{3000}b\\\tc");
        assert_eq!(esc("a*b?", Root, Normal), "a*b?");
        // the index found in the lower case, copied from the text
        assert_eq!(
            esc("\u{130}:", Root, Normal),
            "java.lang.IndexOutOfBoundsException: Range [2, 3) out of bounds for length 2"
        );
        assert_eq!(
            esc("\u{130}a:", Root, Normal),
            "java.lang.IndexOutOfBoundsException: Range [3, 4) out of bounds for length 3"
        );
        // the backslash lands one unit late
        assert_eq!(esc("\u{130}:a", Root, Normal), "\u{130}:\\a");
        assert_eq!(esc("a:\u{130}", Root, Normal), "a\\:\u{130}");
        assert_eq!(esc("\u{130}:", Turkic, Normal), "\u{130}\\:");
        assert_eq!(esc("I\u{307}:", Turkic, Normal), "I\\\u{307}:");
        assert_eq!(
            esc("\u{cc}:", Lithuanian, Normal),
            "java.lang.IndexOutOfBoundsException: Range [0, 3) out of bounds for length 2"
        );
        let escaped = Text {
            units: units("a*?\\"),
            escaped: Some(vec![false, true, false, false]),
        };
        assert_eq!(
            std::string::String::from_utf16_lossy(&escape(&escaped, Root, Normal).unwrap()),
            "a\\*?\\\\"
        );
        assert_eq!(CaseLocale::of_tag("tr-TR"), Turkic);
        assert_eq!(CaseLocale::of_tag("AZ"), Turkic);
        assert_eq!(CaseLocale::of_tag("lt"), Lithuanian);
        assert_eq!(CaseLocale::of_tag("en-US"), Root);
    }
}
