//! Java `String`s as OpenNLP sees them: UTF-16 units, compared, hashed and
//! sliced by unit, so that a feature built from a lone surrogate is the
//! feature Java builds (`String` holds it; Rust's `str` cannot). Features,
//! predicates and outcomes are [`JString`]s; the helpers here are the
//! `java.lang`/`opennlp.tools.util.StringUtil` methods the tools call.

use lucene_analysis::java_character as jc;

/// A Java `String`: its UTF-16 units.
pub type JString = Vec<u16>;

/// `s` as UTF-16 units.
pub fn units(s: &str) -> JString {
    s.encode_utf16().collect()
}

/// The units as a Rust string, a lone surrogate as U+FFFD.
pub fn to_string(units: &[u16]) -> String {
    String::from_utf16_lossy(units)
}

/// `a + b` (Java string concatenation) into a new string.
pub fn concat(parts: &[&[u16]]) -> JString {
    let mut out = Vec::with_capacity(parts.iter().map(|p| p.len()).sum());
    for p in parts {
        out.extend_from_slice(p);
    }
    out
}

/// `StringUtil.isWhitespace(char)`: `Character.isWhitespace` or a space
/// separator (which takes in the no-break spaces Java's `isWhitespace`
/// leaves out).
pub fn is_whitespace(c: u16) -> bool {
    let cp = u32::from(c);
    jc::is_whitespace(cp) || jc::get_type(cp) == jc::SPACE_SEPARATOR
}

/// `Character.isUpperCase(char)`: `Lu`, plus the BMP's `Other_Uppercase`
/// code points (the Roman numerals and the circled capitals).
pub fn is_upper_case(c: u16) -> bool {
    jc::get_type(u32::from(c)) == jc::UPPERCASE_LETTER
        || (0x2160..=0x216F).contains(&c)
        || (0x24B6..=0x24CF).contains(&c)
}

/// `Character.isLetter(char)`.
pub fn is_letter(c: u16) -> bool {
    jc::is_letter(u32::from(c))
}

/// `Character.isDigit(char)`.
pub fn is_digit(c: u16) -> bool {
    jc::is_digit(u32::from(c))
}

/// The code points of `s` (`String.codePoints()`): a lone surrogate is its
/// own code point.
pub fn code_points(s: &[u16]) -> impl Iterator<Item = u32> + '_ {
    char::decode_utf16(s.iter().copied())
        .map(|r| r.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
}

/// Appends code point `cp` as UTF-16 (`Character.toChars`).
pub fn push_code_point(out: &mut JString, cp: u32) {
    match char::from_u32(cp) {
        Some(c) => {
            let mut buf = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut buf));
        }
        // A lone surrogate (the only code points `char` refuses).
        None => out.push(cp as u16),
    }
}

/// `StringUtil.toLowerCase(CharSequence)`: `Character.toLowerCase` per code
/// point, without `String.toLowerCase`'s special cases.
pub fn to_lower_case(s: &[u16]) -> JString {
    let mut out = Vec::with_capacity(s.len());
    for cp in code_points(s) {
        push_code_point(&mut out, jc::to_lower_case(cp));
    }
    out
}

/// `String.toLowerCase()` in a locale without tailorings: Lucene's port of
/// the JDK's (U+0130 and the final sigma included).
pub fn string_to_lower_case(s: &[u16]) -> JString {
    lucene_analysis::lang::java_string_to_lower_case(s)
}

/// `String.trim()`: the units up to U+0020 removed from both ends.
// ARITH: an index within the slice.
#[allow(clippy::arithmetic_side_effects)]
pub fn trim(s: &[u16]) -> &[u16] {
    let start = s.iter().position(|&c| c > 0x20).unwrap_or(s.len());
    let end = s.iter().rposition(|&c| c > 0x20).map_or(start, |i| i + 1);
    &s[start..end]
}

/// `String.indexOf(String, fromIndex)`: the first `needle` at or after
/// `from` (a negative or past-the-end start clamped, as Java's).
// ARITH: an index within the haystack.
#[allow(clippy::arithmetic_side_effects)]
pub fn index_of(hay: &[u16], needle: &[u16], from: usize) -> Option<usize> {
    let from = from.min(hay.len());
    if needle.is_empty() {
        return Some(from);
    }
    hay.get(from..)?
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// `Integer.parseInt(String)`: an optional sign, then decimal digits of any
/// script (`Character.digit(c, 10)`), within `int`.
// ARITH: the value is checked below `2^31 + 1` at every digit.
#[allow(clippy::arithmetic_side_effects)]
pub fn parse_int(s: &[u16]) -> Option<i32> {
    let (negative, digits) = match s.first() {
        Some(&0x2D) => (true, &s[1..]),
        Some(&0x2B) => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut value: i64 = 0;
    for cp in code_points(digits) {
        let d = jc::decimal_digit_value(cp)?;
        value = value.checked_mul(10)?.checked_add(i64::from(d))?;
        if value > i64::from(i32::MAX) + 1 {
            return None;
        }
    }
    let value = if negative { -value } else { value };
    i32::try_from(value).ok()
}

/// `Boolean.parseBoolean(String)`: `true` ignoring ASCII case.
pub fn parse_boolean(s: &str) -> bool {
    s.eq_ignore_ascii_case("true")
}

/// `Integer.toString(int)`.
pub fn int_units(i: usize) -> JString {
    units(&i.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn java_string_helpers() {
        assert!(is_whitespace(0x20) && is_whitespace(0xA0) && is_whitespace(0x2007));
        assert!(!is_whitespace(0x41) && !is_whitespace(0xD800));
        assert!(is_upper_case(0x41) && is_upper_case(0x2160) && is_upper_case(0x24C0));
        assert!(!is_upper_case(0x61));
        assert!(is_letter(0x61) && !is_letter(0x31) && is_digit(0x0663));
        assert_eq!(to_lower_case(&units("ÀBİ")), units("àbi"));
        assert_eq!(string_to_lower_case(&units("İ")), units("i\u{307}"));
        assert_eq!(to_lower_case(&[0xD800, 0x41]), vec![0xD800, 0x61]);
        assert_eq!(to_lower_case(&units("𐐀")), units("𐐨"));
        assert_eq!(trim(&units(" \t a b \n")), units("a b").as_slice());
        assert_eq!(trim(&units("   ")), &[] as &[u16]);
        assert_eq!(index_of(&units("abcabc"), &units("bc"), 2), Some(4));
        assert_eq!(index_of(&units("abc"), &units("x"), 0), None);
        assert_eq!(index_of(&units("abc"), &[], 9), Some(3));
        assert_eq!(index_of(&units("abc"), &units("abcd"), 0), None);
        assert_eq!(parse_int(&units("-42")), Some(-42));
        assert_eq!(parse_int(&units("+7")), Some(7));
        assert_eq!(parse_int(&units("٣")), Some(3));
        assert_eq!(parse_int(&units("2147483647")), Some(i32::MAX));
        assert_eq!(parse_int(&units("-2147483648")), Some(i32::MIN));
        assert_eq!(parse_int(&units("2147483648")), None);
        assert_eq!(parse_int(&units("99999999999999999999")), None);
        assert_eq!(parse_int(&units("-")), None);
        assert_eq!(parse_int(&units("1a")), None);
        assert_eq!(parse_int(&[]), None);
        assert!(parse_boolean("TRUE") && !parse_boolean("yes"));
        assert_eq!(int_units(12), units("12"));
        assert_eq!(concat(&[&units("a"), &units("bc")]), units("abc"));
        assert_eq!(to_string(&[0xD800]), "\u{FFFD}");
        let mut v = Vec::new();
        push_code_point(&mut v, 0xDC00);
        assert_eq!(v, vec![0xDC00]);
    }
}
