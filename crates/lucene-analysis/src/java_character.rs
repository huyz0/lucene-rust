//! The `java.lang.Character` classification analysis-common calls, and
//! Lucene's `org.apache.lucene.analysis.util.UnicodeProps`.
//!
//! Java's tokenizers and filters decide on `Character.getType` (and the
//! predicates built on it: `isLetter`, `isDigit`, `isWhitespace`). Rust's
//! `char` predicates answer different questions -- `char::is_alphabetic` is
//! the `Alphabetic` property, which takes in marks and letter numbers
//! `isLetter` rejects -- so the general category is read from
//! [`java_character_tables`](crate::java_character_tables), generated from
//! the Unicode Character Database 16.0.0 (the version JDK 25 implements and
//! Lucene 10.5.0's `UnicodeProps` names) by
//! `crates/lucene-analysis/tools/gen_java_character_tables.py`. The data is
//! the UCD's, under the Unicode licence (`docs/licences.md`); Java's
//! `Character.getType` is specified as that same `General_Category`, and the
//! tables were checked identical to JDK 25's code point for code point.
//!
//! [`to_upper_case`]/[`to_lower_case`] read `lucene-util`'s table of JDK
//! 25's simple case mappings, the same Unicode 16.0; every case map in this
//! crate goes through them, never through `char`'s (the toolchain's Unicode
//! version). JDK 25 is the target because it is what OpenSearch 3.8.0
//! bundles. CI's fixture JDK is 21 (Unicode 15.0): the analysis-common
//! corpus holds no character whose properties differ between the two, and
//! `codepoints.words` (every code point) leaves those out, so the fixtures
//! agree under either; `tests/code_point_fixtures.rs` pins them to JDK 25.

use std::sync::LazyLock;

use crate::java_character_tables::{
    DECIMAL_DIGIT_ZEROS, GENERAL_CATEGORY_RUNS, UNICODE_WHITESPACE,
};

/// `Character.UNASSIGNED`.
pub const UNASSIGNED: u8 = 0;
/// `Character.UPPERCASE_LETTER`.
pub const UPPERCASE_LETTER: u8 = 1;
/// `Character.LOWERCASE_LETTER`.
pub const LOWERCASE_LETTER: u8 = 2;
/// `Character.TITLECASE_LETTER`.
pub const TITLECASE_LETTER: u8 = 3;
/// `Character.MODIFIER_LETTER`.
pub const MODIFIER_LETTER: u8 = 4;
/// `Character.OTHER_LETTER`.
pub const OTHER_LETTER: u8 = 5;
/// `Character.NON_SPACING_MARK`.
pub const NON_SPACING_MARK: u8 = 6;
/// `Character.ENCLOSING_MARK`.
pub const ENCLOSING_MARK: u8 = 7;
/// `Character.COMBINING_SPACING_MARK`.
pub const COMBINING_SPACING_MARK: u8 = 8;
/// `Character.DECIMAL_DIGIT_NUMBER`.
pub const DECIMAL_DIGIT_NUMBER: u8 = 9;
/// `Character.LETTER_NUMBER`.
pub const LETTER_NUMBER: u8 = 10;
/// `Character.OTHER_NUMBER`.
pub const OTHER_NUMBER: u8 = 11;
/// `Character.SPACE_SEPARATOR`.
pub const SPACE_SEPARATOR: u8 = 12;
/// `Character.LINE_SEPARATOR`.
pub const LINE_SEPARATOR: u8 = 13;
/// `Character.PARAGRAPH_SEPARATOR`.
pub const PARAGRAPH_SEPARATOR: u8 = 14;
/// `Character.CONTROL`.
pub const CONTROL: u8 = 15;
/// `Character.FORMAT`.
pub const FORMAT: u8 = 16;
/// `Character.PRIVATE_USE`.
pub const PRIVATE_USE: u8 = 18;
/// `Character.SURROGATE`.
pub const SURROGATE: u8 = 19;
/// `Character.DASH_PUNCTUATION`.
pub const DASH_PUNCTUATION: u8 = 20;
/// `Character.START_PUNCTUATION`.
pub const START_PUNCTUATION: u8 = 21;
/// `Character.END_PUNCTUATION`.
pub const END_PUNCTUATION: u8 = 22;
/// `Character.CONNECTOR_PUNCTUATION`.
pub const CONNECTOR_PUNCTUATION: u8 = 23;
/// `Character.OTHER_PUNCTUATION`.
pub const OTHER_PUNCTUATION: u8 = 24;
/// `Character.MATH_SYMBOL`.
pub const MATH_SYMBOL: u8 = 25;
/// `Character.CURRENCY_SYMBOL`.
pub const CURRENCY_SYMBOL: u8 = 26;
/// `Character.MODIFIER_SYMBOL`.
pub const MODIFIER_SYMBOL: u8 = 27;
/// `Character.OTHER_SYMBOL`.
pub const OTHER_SYMBOL: u8 = 28;
/// `Character.INITIAL_QUOTE_PUNCTUATION`.
pub const INITIAL_QUOTE_PUNCTUATION: u8 = 29;
/// `Character.FINAL_QUOTE_PUNCTUATION`.
pub const FINAL_QUOTE_PUNCTUATION: u8 = 30;

/// `Character.MAX_CODE_POINT`.
pub const MAX_CODE_POINT: u32 = 0x10FFFF;

/// Latin-1's categories, the hot range, read once from the runs.
static LATIN1: std::sync::LazyLock<[u8; 256]> = std::sync::LazyLock::new(|| {
    let mut t = [0u8; 256];
    for (cp, slot) in t.iter_mut().enumerate() {
        *slot = lookup(cp as u32);
    }
    t
});

fn lookup(cp: u32) -> u8 {
    let i = GENERAL_CATEGORY_RUNS.partition_point(|&(start, _)| start <= cp);
    // Run 0 starts at code point 0, so `i >= 1` for every `cp`.
    GENERAL_CATEGORY_RUNS[i.saturating_sub(1)].1
}

/// `Character.getType(int)`; `UNASSIGNED` past [`MAX_CODE_POINT`].
pub fn get_type(cp: u32) -> u8 {
    if cp < 256 {
        LATIN1[cp as usize]
    } else if cp > MAX_CODE_POINT {
        UNASSIGNED
    } else {
        lookup(cp)
    }
}

/// `Character.isLetter(int)`: categories `Lu Ll Lt Lm Lo`.
pub fn is_letter(cp: u32) -> bool {
    matches!(
        get_type(cp),
        UPPERCASE_LETTER | LOWERCASE_LETTER | TITLECASE_LETTER | MODIFIER_LETTER | OTHER_LETTER
    )
}

/// `Character.isDigit(int)`: category `Nd`.
pub fn is_digit(cp: u32) -> bool {
    get_type(cp) == DECIMAL_DIGIT_NUMBER
}

/// `Character.isLetterOrDigit(int)`.
pub fn is_letter_or_digit(cp: u32) -> bool {
    is_letter(cp) || is_digit(cp)
}

/// `Character.isWhitespace(int)`: a space, line or paragraph separator
/// other than the three non-breaking spaces, or one of `\t \n \u000B \f \r
/// \u001C..\u001F`.
pub fn is_whitespace(cp: u32) -> bool {
    if cp < 0x80 {
        return matches!(cp, 0x09..=0x0D | 0x1C..=0x20);
    }
    match cp {
        0x00A0 | 0x2007 | 0x202F => false,
        _ => matches!(
            get_type(cp),
            SPACE_SEPARATOR | LINE_SEPARATOR | PARAGRAPH_SEPARATOR
        ),
    }
}

/// `Character.isLowerCase(int)` restricted to the general category
/// (`Ll`); Java also counts `Other_Lowercase` code points, which no caller
/// here distinguishes.
pub fn is_lower_case_letter(cp: u32) -> bool {
    get_type(cp) == LOWERCASE_LETTER
}

/// Bit `cp` set for every BMP decimal digit (`Nd`), so that
/// [`decimal_digit_value`] of a BMP non-digit is one load.
static BMP_DIGITS: LazyLock<[u64; 1024]> = LazyLock::new(|| {
    let mut bits = [0u64; 1024];
    for zero in DECIMAL_DIGIT_ZEROS.iter().filter(|&&z| z < 0x10000) {
        for cp in *zero..zero + 10 {
            if is_digit(cp) {
                bits[(cp >> 6) as usize] |= 1 << (cp & 63);
            }
        }
    }
    bits
});

/// `Character.getNumericValue(int)` of a decimal digit (`Nd`): its value
/// 0..=9; `None` for any other code point.
pub fn decimal_digit_value(cp: u32) -> Option<u32> {
    // A BMP code point is first looked up in a bitmap of the plane's digits.
    if cp < 0x10000 && BMP_DIGITS[(cp >> 6) as usize] & (1 << (cp & 63)) == 0 {
        return None;
    }
    // A digit always has its run's zero at or below it, within nine.
    let i = DECIMAL_DIGIT_ZEROS.partition_point(|&z| z <= cp);
    let zero = DECIMAL_DIGIT_ZEROS[i.checked_sub(1)?];
    (cp - zero < 10 && is_digit(cp)).then(|| cp - zero)
}

/// `Character.toUpperCase(int)`: the JDK's simple uppercase mapping.
#[inline]
pub fn to_upper_case(cp: u32) -> u32 {
    lucene_util::automaton::java_to_upper_case(cp as i32) as u32
}

/// `Character.toLowerCase(int)`: the JDK's simple lowercase mapping.
#[inline]
pub fn to_lower_case(cp: u32) -> u32 {
    lucene_util::automaton::java_to_lower_case(cp as i32) as u32
}

/// `UnicodeProps.WHITESPACE.get(int)`: Unicode's `White_Space` property as
/// Lucene 10.5.0 ships it (which, unlike [`is_whitespace`], includes the
/// non-breaking spaces and U+0085).
pub fn is_unicode_whitespace(cp: u32) -> bool {
    let i = UNICODE_WHITESPACE.partition_point(|&(start, _)| start <= cp);
    i > 0 && cp <= UNICODE_WHITESPACE[i - 1].1
}

/// A UTF-16 unit is a high (leading) surrogate.
#[inline]
pub fn is_high_surrogate(u: u16) -> bool {
    (0xD800..=0xDBFF).contains(&u)
}

/// A UTF-16 unit is a low (trailing) surrogate.
#[inline]
pub fn is_low_surrogate(u: u16) -> bool {
    (0xDC00..=0xDFFF).contains(&u)
}

/// `Character.codePointAt(char[] a, int index, int limit)`.
#[inline]
pub fn code_point_at(buf: &[u16], index: usize, limit: usize) -> u32 {
    let hi = buf[index];
    if is_high_surrogate(hi) && index + 1 < limit {
        let lo = buf[index + 1];
        if is_low_surrogate(lo) {
            return 0x10000 + ((u32::from(hi) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
        }
    }
    u32::from(hi)
}

/// `Character.charCount(int)`.
#[inline]
pub fn char_count(cp: u32) -> usize {
    if cp >= 0x10000 {
        2
    } else {
        1
    }
}

/// `Character.toChars(int, char[], int)`: appends `cp`'s UTF-16 units,
/// returning how many.
#[inline]
pub fn push_utf16(out: &mut Vec<u16>, cp: u32) -> usize {
    if cp >= 0x10000 {
        let v = cp - 0x10000;
        out.push(0xD800 | (v >> 10) as u16);
        out.push(0xDC00 | (v & 0x3FF) as u16);
        2
    } else {
        out.push(cp as u16);
        1
    }
}

/// The full uppercase mappings `String.toUpperCase(Locale.ROOT)` applies
/// beyond [`to_upper_case`]: the unconditional multi-character mappings of
/// Unicode's `SpecialCasing.txt` (`ß` -> `SS`, `ŉ` -> `ʼN`, the Greek
/// iota-subscript forms ...), as `(char, [up to three UTF-16 units, 0-padded])`
/// sorted by char. Every one is a BMP character mapping to two or three BMP
/// units. Read off JDK 25 and JDK 21 (`String.valueOf(c).toUpperCase(ROOT)` for
/// every code point; the two lists are identical, and every other code
/// point's full mapping is its simple one).
pub static SPECIAL_UPPER: [(u16, [u16; 3]); 102] = [
    (0x00DF, [0x0053, 0x0053, 0x0]),
    (0x0149, [0x02BC, 0x004E, 0x0]),
    (0x01F0, [0x004A, 0x030C, 0x0]),
    (0x0390, [0x0399, 0x0308, 0x0301]),
    (0x03B0, [0x03A5, 0x0308, 0x0301]),
    (0x0587, [0x0535, 0x0552, 0x0]),
    (0x1E96, [0x0048, 0x0331, 0x0]),
    (0x1E97, [0x0054, 0x0308, 0x0]),
    (0x1E98, [0x0057, 0x030A, 0x0]),
    (0x1E99, [0x0059, 0x030A, 0x0]),
    (0x1E9A, [0x0041, 0x02BE, 0x0]),
    (0x1F50, [0x03A5, 0x0313, 0x0]),
    (0x1F52, [0x03A5, 0x0313, 0x0300]),
    (0x1F54, [0x03A5, 0x0313, 0x0301]),
    (0x1F56, [0x03A5, 0x0313, 0x0342]),
    (0x1F80, [0x1F08, 0x0399, 0x0]),
    (0x1F81, [0x1F09, 0x0399, 0x0]),
    (0x1F82, [0x1F0A, 0x0399, 0x0]),
    (0x1F83, [0x1F0B, 0x0399, 0x0]),
    (0x1F84, [0x1F0C, 0x0399, 0x0]),
    (0x1F85, [0x1F0D, 0x0399, 0x0]),
    (0x1F86, [0x1F0E, 0x0399, 0x0]),
    (0x1F87, [0x1F0F, 0x0399, 0x0]),
    (0x1F88, [0x1F08, 0x0399, 0x0]),
    (0x1F89, [0x1F09, 0x0399, 0x0]),
    (0x1F8A, [0x1F0A, 0x0399, 0x0]),
    (0x1F8B, [0x1F0B, 0x0399, 0x0]),
    (0x1F8C, [0x1F0C, 0x0399, 0x0]),
    (0x1F8D, [0x1F0D, 0x0399, 0x0]),
    (0x1F8E, [0x1F0E, 0x0399, 0x0]),
    (0x1F8F, [0x1F0F, 0x0399, 0x0]),
    (0x1F90, [0x1F28, 0x0399, 0x0]),
    (0x1F91, [0x1F29, 0x0399, 0x0]),
    (0x1F92, [0x1F2A, 0x0399, 0x0]),
    (0x1F93, [0x1F2B, 0x0399, 0x0]),
    (0x1F94, [0x1F2C, 0x0399, 0x0]),
    (0x1F95, [0x1F2D, 0x0399, 0x0]),
    (0x1F96, [0x1F2E, 0x0399, 0x0]),
    (0x1F97, [0x1F2F, 0x0399, 0x0]),
    (0x1F98, [0x1F28, 0x0399, 0x0]),
    (0x1F99, [0x1F29, 0x0399, 0x0]),
    (0x1F9A, [0x1F2A, 0x0399, 0x0]),
    (0x1F9B, [0x1F2B, 0x0399, 0x0]),
    (0x1F9C, [0x1F2C, 0x0399, 0x0]),
    (0x1F9D, [0x1F2D, 0x0399, 0x0]),
    (0x1F9E, [0x1F2E, 0x0399, 0x0]),
    (0x1F9F, [0x1F2F, 0x0399, 0x0]),
    (0x1FA0, [0x1F68, 0x0399, 0x0]),
    (0x1FA1, [0x1F69, 0x0399, 0x0]),
    (0x1FA2, [0x1F6A, 0x0399, 0x0]),
    (0x1FA3, [0x1F6B, 0x0399, 0x0]),
    (0x1FA4, [0x1F6C, 0x0399, 0x0]),
    (0x1FA5, [0x1F6D, 0x0399, 0x0]),
    (0x1FA6, [0x1F6E, 0x0399, 0x0]),
    (0x1FA7, [0x1F6F, 0x0399, 0x0]),
    (0x1FA8, [0x1F68, 0x0399, 0x0]),
    (0x1FA9, [0x1F69, 0x0399, 0x0]),
    (0x1FAA, [0x1F6A, 0x0399, 0x0]),
    (0x1FAB, [0x1F6B, 0x0399, 0x0]),
    (0x1FAC, [0x1F6C, 0x0399, 0x0]),
    (0x1FAD, [0x1F6D, 0x0399, 0x0]),
    (0x1FAE, [0x1F6E, 0x0399, 0x0]),
    (0x1FAF, [0x1F6F, 0x0399, 0x0]),
    (0x1FB2, [0x1FBA, 0x0399, 0x0]),
    (0x1FB3, [0x0391, 0x0399, 0x0]),
    (0x1FB4, [0x0386, 0x0399, 0x0]),
    (0x1FB6, [0x0391, 0x0342, 0x0]),
    (0x1FB7, [0x0391, 0x0342, 0x0399]),
    (0x1FBC, [0x0391, 0x0399, 0x0]),
    (0x1FC2, [0x1FCA, 0x0399, 0x0]),
    (0x1FC3, [0x0397, 0x0399, 0x0]),
    (0x1FC4, [0x0389, 0x0399, 0x0]),
    (0x1FC6, [0x0397, 0x0342, 0x0]),
    (0x1FC7, [0x0397, 0x0342, 0x0399]),
    (0x1FCC, [0x0397, 0x0399, 0x0]),
    (0x1FD2, [0x0399, 0x0308, 0x0300]),
    (0x1FD3, [0x0399, 0x0308, 0x0301]),
    (0x1FD6, [0x0399, 0x0342, 0x0]),
    (0x1FD7, [0x0399, 0x0308, 0x0342]),
    (0x1FE2, [0x03A5, 0x0308, 0x0300]),
    (0x1FE3, [0x03A5, 0x0308, 0x0301]),
    (0x1FE4, [0x03A1, 0x0313, 0x0]),
    (0x1FE6, [0x03A5, 0x0342, 0x0]),
    (0x1FE7, [0x03A5, 0x0308, 0x0342]),
    (0x1FF2, [0x1FFA, 0x0399, 0x0]),
    (0x1FF3, [0x03A9, 0x0399, 0x0]),
    (0x1FF4, [0x038F, 0x0399, 0x0]),
    (0x1FF6, [0x03A9, 0x0342, 0x0]),
    (0x1FF7, [0x03A9, 0x0342, 0x0399]),
    (0x1FFC, [0x03A9, 0x0399, 0x0]),
    (0xFB00, [0x0046, 0x0046, 0x0]),
    (0xFB01, [0x0046, 0x0049, 0x0]),
    (0xFB02, [0x0046, 0x004C, 0x0]),
    (0xFB03, [0x0046, 0x0046, 0x0049]),
    (0xFB04, [0x0046, 0x0046, 0x004C]),
    (0xFB05, [0x0053, 0x0054, 0x0]),
    (0xFB06, [0x0053, 0x0054, 0x0]),
    (0xFB13, [0x0544, 0x0546, 0x0]),
    (0xFB14, [0x0544, 0x0535, 0x0]),
    (0xFB15, [0x0544, 0x053B, 0x0]),
    (0xFB16, [0x054E, 0x0546, 0x0]),
    (0xFB17, [0x0544, 0x053D, 0x0]),
];

/// [`SPECIAL_UPPER`]'s mapping of a UTF-16 unit, if it has one.
pub fn special_upper(unit: u16) -> Option<&'static [u16]> {
    let i = SPECIAL_UPPER
        .binary_search_by_key(&unit, |&(c, _)| c)
        .ok()?;
    let m = &SPECIAL_UPPER[i].1;
    Some(if m[2] == 0 { &m[..2] } else { &m[..] })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The case table (JDK 25) and the category table (UCD 16.0) are the
    /// same Unicode version: no code point has a case mapping and no
    /// category.
    #[test]
    fn case_mappings_and_categories_agree() {
        for cp in 0..=0x10_FFFF {
            if to_lower_case(cp) != cp || to_upper_case(cp) != cp {
                assert_ne!(get_type(cp), UNASSIGNED, "{cp:#x}");
            }
        }
    }

    #[test]
    fn categories_match_java() {
        assert_eq!(get_type('A' as u32), UPPERCASE_LETTER);
        assert_eq!(get_type('a' as u32), LOWERCASE_LETTER);
        assert_eq!(get_type('5' as u32), DECIMAL_DIGIT_NUMBER);
        assert_eq!(get_type(' ' as u32), SPACE_SEPARATOR);
        assert_eq!(get_type(0x2028), LINE_SEPARATOR);
        assert_eq!(get_type(0x0301), NON_SPACING_MARK);
        assert_eq!(get_type(0x4E2D), OTHER_LETTER);
        assert_eq!(get_type(0x1F600), OTHER_SYMBOL);
        assert_eq!(get_type(0xD800), SURROGATE);
        assert_eq!(get_type(0x110000), UNASSIGNED);
        assert_eq!(get_type(0x10FFFF), UNASSIGNED);
        assert_eq!(get_type(0xE000), PRIVATE_USE);
    }

    #[test]
    fn predicates() {
        assert!(is_letter('é' as u32));
        assert!(!is_letter(0x0301));
        assert!(!is_letter(0x2160), "Roman numeral one is Nl, not a letter");
        assert!(is_digit(0x0660));
        assert!(is_letter_or_digit('7' as u32));
        assert!(is_whitespace('\t' as u32));
        assert!(is_whitespace(0x1F));
        assert!(is_whitespace(0x3000));
        assert!(!is_whitespace(0x00A0));
        assert!(!is_whitespace(0x2007));
        assert!(!is_whitespace(0x202F));
        assert!(!is_whitespace(0x0085));
        assert!(is_unicode_whitespace(0x00A0));
        assert!(is_unicode_whitespace(0x0085));
        assert!(!is_unicode_whitespace('x' as u32));
        assert!(!is_unicode_whitespace(0x1C));
        assert!(is_lower_case_letter('q' as u32));
    }

    #[test]
    fn special_upper_mappings() {
        assert_eq!(special_upper(0xDF), Some(&[0x53u16, 0x53][..]));
        assert_eq!(special_upper(0x390), Some(&[0x399u16, 0x308, 0x301][..]));
        assert_eq!(special_upper('a' as u16), None);
        assert!(SPECIAL_UPPER.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn digits_and_case() {
        assert_eq!(decimal_digit_value('7' as u32), Some(7));
        assert_eq!(decimal_digit_value(0x0669), Some(9));
        assert_eq!(decimal_digit_value(0x1D7CE), Some(0));
        assert_eq!(decimal_digit_value('x' as u32), None);
        for cp in 0..=0x10FFFF {
            assert_eq!(decimal_digit_value(cp).is_some(), is_digit(cp), "{cp:X}");
        }
        assert_eq!(to_upper_case('ß' as u32), 'ß' as u32);
        assert_eq!(to_upper_case('a' as u32), 'A' as u32);
        assert_eq!(to_lower_case(0x0130), 'i' as u32);
    }

    #[test]
    fn utf16_helpers() {
        let buf: Vec<u16> = "a😀".encode_utf16().collect();
        assert_eq!(code_point_at(&buf, 0, 3), 'a' as u32);
        assert_eq!(code_point_at(&buf, 1, 3), 0x1F600);
        assert_eq!(code_point_at(&buf, 1, 2), 0xD83D, "limit cuts the pair");
        assert_eq!(code_point_at(&buf, 2, 3), 0xDE00);
        assert_eq!(char_count(0x1F600), 2);
        assert_eq!(char_count(0x41), 1);
        let mut out = Vec::new();
        assert_eq!(push_utf16(&mut out, 0x1F600), 2);
        assert_eq!(push_utf16(&mut out, 0x41), 1);
        assert_eq!(&out[..2], &buf[1..]);
        assert!(is_high_surrogate(0xD83D) && !is_high_surrogate(0xDE00));
        assert!(is_low_surrogate(0xDE00) && !is_low_surrogate(0xD83D));
    }
}
