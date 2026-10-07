//! The final-sigma condition of `String.toLowerCase` (the JDK's
//! `ConditionalSpecialCasing`): a capital sigma lowercases to `ς` when a
//! cased letter precedes it and none follows it, both looked for only
//! within the word holding it -- between the boundaries of
//! `BreakIterator.getWordInstance()`, the JDK's legacy rule-based word
//! iterator, not UAX #29 and not Unicode's `Case_Ignorable` context.
//!
//! The word boundaries are re-specified here from black-box runs of the JDK
//! (no JDK code is copied, `docs/licences.md`): a word token is the longest
//! of
//!
//! - letters (`L*`, `Mc`, Thai's dictionary range; not kanji, katakana or
//!   hiragana), joined across single mid-word characters (`Pd`, `Pc`,
//!   U+00AD, U+2027, `"`, `'`, `.`), then an optional danda, alternating
//!   with numbers (`N*` digits joined across single `"`, `'`, `,`, U+066B,
//!   `.`), the last number optionally followed by `%`, `&`, U+00A2, U+066A,
//!   U+2030 or U+2031;
//! - the same led by a currency symbol, `#`, `.` or U+00A2;
//! - a run of spaces and tabs, then an optional CR and line separator;
//! - a run of katakana, of hiragana, of kanji or of Thai;
//! - any one character.
//!
//! Non-spacing and enclosing marks and format characters (but U+00AD) are
//! skipped after a letter or digit; only format characters are skipped
//! after a mid-word, mid-number, pre- or post-number character or a danda,
//! and at the start of a token (a mark there is a token of its own).
//! Matching that, the JDK's `isBoundary(i)` also answers `true` right after
//! a supplementary character that does not start the text, where iterating
//! the boundaries does not stop.
//!
//! These rules give the JDK's boundaries for every BMP code point between
//! letters, digits and itself, but one: U+00A2 between a letter and a
//! digit (`a¢1`) is a boundary on both sides in the JDK, here only before.
//!
//! `isCased` is the JDK's: `Lu`, `Ll`, `Lt`, and its list of
//! `Other_Lowercase`/`Other_Uppercase` ranges.
//!
//! Checked against `String.toLowerCase(Locale.ROOT)` by
//! `GenAnalysisLanguages.java`'s `final_sigma.txt` (random strings over
//! every class above, identical under JDK 21 and 25).

use crate::java_character::{self as jc, get_type};

fn format(c: u32) -> bool {
    get_type(c) == jc::FORMAT && c != 0xAD
}

fn ignorable(c: u32) -> bool {
    matches!(get_type(c), jc::NON_SPACING_MARK | jc::ENCLOSING_MARK) || format(c)
}

fn kanji(c: u32) -> bool {
    c == 0x3005 || (0x4E00..=0x9FA5).contains(&c) || (0xF900..=0xFA2D).contains(&c)
}

fn katakana(c: u32) -> bool {
    (0x3099..=0x309C).contains(&c) || (0x30A1..=0x30FE).contains(&c)
}

fn hiragana(c: u32) -> bool {
    (0x3041..=0x3094).contains(&c) || (0x309D..=0x309E).contains(&c) || c == 0x30FC
}

fn thai(c: u32) -> bool {
    matches!(c, 0x0E01..=0x0E2E | 0x0E30..=0x0E3A | 0x0E40..=0x0E44 | 0x0E47..=0x0E4E)
        && !ignorable(c)
}

fn letter(c: u32) -> bool {
    let t = get_type(c);
    (matches!(
        t,
        jc::UPPERCASE_LETTER
            | jc::LOWERCASE_LETTER
            | jc::TITLECASE_LETTER
            | jc::MODIFIER_LETTER
            | jc::OTHER_LETTER
            | jc::COMBINING_SPACING_MARK
    ) || thai(c))
        && !(kanji(c) || katakana(c) || hiragana(c) || ignorable(c))
}

fn digit(c: u32) -> bool {
    matches!(
        get_type(c),
        jc::DECIMAL_DIGIT_NUMBER | jc::LETTER_NUMBER | jc::OTHER_NUMBER
    )
}

fn mid_word(c: u32) -> bool {
    matches!(
        get_type(c),
        jc::DASH_PUNCTUATION | jc::CONNECTOR_PUNCTUATION
    ) || matches!(c, 0xAD | 0x2027 | 0x22 | 0x27 | 0x2E)
}

fn mid_number(c: u32) -> bool {
    matches!(c, 0x22 | 0x27 | 0x2C | 0x066B | 0x2E)
}

fn pre_number(c: u32) -> bool {
    get_type(c) == jc::CURRENCY_SYMBOL || matches!(c, 0x23 | 0x2E | 0xA2)
}

fn post_number(c: u32) -> bool {
    matches!(c, 0x25 | 0x26 | 0xA2 | 0x066A | 0x2030 | 0x2031)
}

fn danda(c: u32) -> bool {
    matches!(c, 0x0964 | 0x0965)
}

fn space(c: u32) -> bool {
    get_type(c) == jc::SPACE_SEPARATOR || c == 0x09
}

fn line_separator(c: u32) -> bool {
    matches!(c, 0x0A | 0x0C | 0x2028 | 0x2029)
}

/// The JDK's `ConditionalSpecialCasing.isCased`.
fn cased(c: u32) -> bool {
    matches!(
        get_type(c),
        jc::UPPERCASE_LETTER | jc::LOWERCASE_LETTER | jc::TITLECASE_LETTER
    ) || matches!(
        c,
        0x02B0..=0x02B8
            | 0x02C0..=0x02C1
            | 0x02E0..=0x02E4
            | 0x0345
            | 0x037A
            | 0x1D2C..=0x1D61
            | 0x2160..=0x217F
            | 0x24B6..=0x24E9
    )
}

/// One word-iterator scan over code points.
struct Words<'a>(&'a [u32]);

impl Words<'_> {
    fn at(&self, i: usize, class: impl Fn(u32) -> bool) -> bool {
        self.0.get(i).is_some_and(|&c| class(c))
    }

    fn skip(&self, mut i: usize, class: impl Fn(u32) -> bool) -> usize {
        while self.at(i, &class) {
            i += 1;
        }
        i
    }

    /// `class+ (mid class+)*`, marks and format characters skipped after
    /// each `class` character, format characters after each `mid`.
    fn run(
        &self,
        i: usize,
        class: impl Fn(u32) -> bool,
        mid: impl Fn(u32) -> bool,
    ) -> Option<usize> {
        if !self.at(i, &class) {
            return None;
        }
        let mut i = self.skip(i + 1, ignorable);
        while self.at(i, &class) {
            i = self.skip(i + 1, ignorable);
        }
        while self.at(i, &mid) {
            let j = self.skip(i + 1, format);
            if !self.at(j, &class) {
                break;
            }
            i = self.skip(j + 1, ignorable);
            while self.at(i, &class) {
                i = self.skip(i + 1, ignorable);
            }
        }
        Some(i)
    }

    /// A word: letters, then an optional danda.
    fn word(&self, i: usize) -> Option<usize> {
        let end = self.run(i, letter, mid_word)?;
        Some(if self.at(end, danda) {
            self.skip(end + 1, format)
        } else {
            end
        })
    }

    fn number(&self, i: usize) -> Option<usize> {
        self.run(i, digit, mid_number)
    }

    /// The longest `(number word)* (number post_number?)?` from `i`.
    fn tail(&self, mut i: usize) -> usize {
        let mut best = i;
        while let Some(n) = self.number(i) {
            best = best.max(n);
            if self.at(n, post_number) {
                best = best.max(self.skip(n + 1, format));
            }
            let Some(w) = self.word(n) else {
                break;
            };
            best = best.max(w);
            i = w;
        }
        best
    }

    /// The end of the token starting at `i` (`i < len`).
    fn token(&self, i: usize) -> usize {
        let s = self.0;
        // Format characters open no token of their own.
        let i = self.skip(i, format);
        if i == s.len() {
            return i;
        }
        let c = s[i];
        let mut end = i + 1;
        if space(c) || c == 0x0D || line_separator(c) {
            let mut j = self.skip(i, space);
            if self.at(j, |c| c == 0x0D) {
                j += 1;
            }
            if self.at(j, line_separator) {
                j += 1;
            }
            end = end.max(j);
        }
        if let Some(w) = self.word(i) {
            end = end.max(self.tail(w));
        }
        if digit(c) {
            end = end.max(self.tail(i));
        }
        if pre_number(c) {
            let k = self.skip(i + 1, format);
            let t = self.tail(k);
            if t > k {
                end = end.max(t);
            }
        }
        for class in [katakana, hiragana, kanji, thai] {
            if class(c) {
                let mut j = i;
                while self.at(j, class) {
                    j = self.skip(j + 1, ignorable);
                }
                end = end.max(j);
            }
        }
        end
    }

    /// Whether each position `0..=len` is a boundary.
    fn boundaries(&self) -> Vec<bool> {
        let mut b = vec![false; self.0.len() + 1];
        b[0] = true;
        let mut i = 0;
        while i < self.0.len() {
            i = self.token(i);
            b[i] = true;
        }
        b
    }
}

/// The final-sigma context for each capital sigma of `cps`: whether the one
/// at code point index `i` lowercases to `ς`. `None` for a text without one.
pub(crate) struct FinalSigma {
    cps: Vec<u32>,
    bounds: Vec<bool>,
}

impl FinalSigma {
    pub(crate) fn new(cps: &[u32]) -> Self {
        FinalSigma {
            cps: cps.to_vec(),
            bounds: Words(cps).boundaries(),
        }
    }

    /// `isBoundary`, with the JDK's answer after a supplementary character.
    fn is_boundary(&self, i: usize) -> bool {
        self.bounds[i] || (i >= 2 && self.cps[i - 1] > 0xFFFF)
    }

    /// `ConditionalSpecialCasing.isFinalCased` at code point index `i`.
    pub(crate) fn is_final(&self, i: usize) -> bool {
        let mut j = i;
        while !self.is_boundary(j) {
            if cased(self.cps[j - 1]) {
                let mut k = i + 1;
                while k < self.cps.len() && !self.is_boundary(k) {
                    if cased(self.cps[k]) {
                        return false;
                    }
                    k += 1;
                }
                return true;
            }
            j -= 1;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finals(s: &str) -> String {
        let cps: Vec<u32> = s.chars().map(u32::from).collect();
        let f = FinalSigma::new(&cps);
        cps.iter()
            .enumerate()
            .filter(|&(_, &c)| c == 0x3A3)
            .map(|(i, _)| if f.is_final(i) { 'F' } else { 's' })
            .collect()
    }

    #[test]
    fn the_reviewers_cases_and_the_word_rules() {
        // A word's cased letters count, not just the neighbours.
        assert_eq!(finals("a1Σ"), "F");
        assert_eq!(finals("aΣ1"), "F");
        assert_eq!(finals("a1Σb"), "s");
        assert_eq!(finals("ħΣ\u{200E}\u{200F}Σ"), "sF");
        assert_eq!(finals("Σ'ǘ"), "s");
        assert_eq!(finals("aΣ'ǘ"), "s");
        assert_eq!(finals("aΣ'"), "F");
        assert_eq!(finals("Σ\u{200C}𐐀"), "s");
        // Word boundaries end the search.
        assert_eq!(finals("a Σ"), "s");
        assert_eq!(finals("a:Σ"), "s");
        assert_eq!(finals("a日Σ"), "s");
        assert_eq!(finals("a-Σ"), "F");
        assert_eq!(finals("aΣ-b"), "s");
        assert_eq!(finals("a.1Σ"), "s");
        // The JDK's `isCased` ranges.
        assert_eq!(finals("ʰΣ"), "F");
        assert_eq!(finals("Ⅱ\u{3A3}"), "F");
        assert_eq!(finals("a\u{345}Σ"), "F");
        // Marks after a mid-word character end the word; format characters
        // do not.
        assert_eq!(finals("aΣ\u{AD}\u{FE0F}ħ"), "F");
        assert_eq!(finals("aΣ\u{2027}\u{200D}A"), "s");
        // After a supplementary letter `isBoundary` answers true.
        assert_eq!(finals("ǘ𐐀Σ"), "s");
        assert_eq!(finals("𐐨Σ"), "F");
        // Number tails, pre- and post-number characters.
        assert_eq!(finals("Σ%ͅ"), "s");
        assert_eq!(finals("aΣ2%\u{345}"), "F");
        assert_eq!(finals("¢\u{300}\u{345}2Σ"), "s");
        assert_eq!(finals("$1aΣ"), "F");
        // Spaces, line ends and the kana, kanji and Thai runs.
        assert_eq!(finals("\u{20}\u{9}\r\nΣ"), "s");
        assert_eq!(finals("アイΣ"), "s");
        assert_eq!(finals("あいΣ"), "s");
        assert_eq!(finals("日本Σ"), "s");
        assert_eq!(finals("aกΣ"), "F");
        assert_eq!(finals("aΣ।\u{345}"), "F");
        // A leading mark is a token; leading format characters are not.
        assert_eq!(finals("\u{345}Σ"), "s");
        assert_eq!(finals("\u{200E}\u{345}Σ"), "s");
        assert_eq!(finals("\u{200E}"), "");
    }
}
