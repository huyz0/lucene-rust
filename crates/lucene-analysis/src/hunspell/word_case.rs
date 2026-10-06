//! `WordCase`: the capitalization class of a word, over UTF-16 units as
//! Java's `Character.isUpperCase(char)`/`isLowerCase(char)` see them.

use crate::java_character as jc;

/// `WordCase`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WordCase {
    /// `WORD`.
    Upper,
    /// `Word`.
    Title,
    /// `word`.
    Lower,
    /// `WoRd` or `wOrd`.
    Mixed,
    /// `-`, `/`, `42`.
    Neutral,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CharCase {
    Upper,
    Lower,
    Neutral,
}

/// `Character.isUpperCase(char)` (the general category; `Other_Uppercase`
/// letters outside it are not distinguished by any dictionary).
pub(crate) fn is_upper(c: u16) -> bool {
    jc::get_type(u32::from(c)) == jc::UPPERCASE_LETTER
}

/// `Character.isLowerCase(char)` (the general category, as [`is_upper`]).
pub(crate) fn is_lower(c: u16) -> bool {
    jc::is_lower_case_letter(u32::from(c))
}

/// `Character.toUpperCase(char)`: a BMP mapping that leaves the BMP is the
/// unit itself, as Java's `char` overload.
pub(crate) fn to_upper(c: u16) -> u16 {
    u16::try_from(jc::to_upper_case(u32::from(c))).unwrap_or(c)
}

/// `Character.toLowerCase(char)`.
pub(crate) fn to_lower(c: u16) -> u16 {
    if c < 0x80 {
        return u16::from((c as u8).to_ascii_lowercase());
    }
    u16::try_from(jc::to_lower_case(u32::from(c))).unwrap_or(c)
}

fn char_case(c: u16) -> CharCase {
    // ASCII, where Java's answer is the letter ranges.
    if c < 0x80 {
        return match c as u8 {
            b'A'..=b'Z' => CharCase::Upper,
            b'a'..=b'z' => CharCase::Lower,
            _ => CharCase::Neutral,
        };
    }
    if is_upper(c) {
        CharCase::Upper
    } else if is_lower(c) && to_upper(c) != c {
        CharCase::Lower
    } else {
        CharCase::Neutral
    }
}

impl WordCase {
    /// `WordCase.caseOf(word, length)` (a non-empty word).
    pub(crate) fn case_of(word: &[u16]) -> WordCase {
        let start = char_case(word[0]);
        let (mut seen_upper, mut seen_lower) = (false, false);
        for &c in &word[1..] {
            let cc = char_case(c);
            seen_upper = seen_upper || cc == CharCase::Upper;
            seen_lower = seen_lower || cc == CharCase::Lower;
            if seen_upper && seen_lower {
                break;
            }
        }
        if seen_lower && seen_upper {
            return WordCase::Mixed;
        }
        match start {
            CharCase::Lower => {
                if seen_upper {
                    WordCase::Mixed
                } else {
                    WordCase::Lower
                }
            }
            CharCase::Upper => {
                if !seen_lower {
                    WordCase::Upper
                } else {
                    WordCase::Title
                }
            }
            CharCase::Neutral => {
                if seen_lower {
                    WordCase::Lower
                } else if seen_upper {
                    WordCase::Upper
                } else {
                    WordCase::Neutral
                }
            }
        }
    }
}
