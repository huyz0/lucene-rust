//! `org.apache.lucene.analysis.hu`: `HungarianAnalyzer` (Snowball) and the
//! light stemmer.

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::ends_with;
use crate::CharArraySet;

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop, CharStemmer, StemFilter};

/// `HungarianAnalyzer.getDefaultStopSet()` (`snowball/hungarian_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/hungarian_stop.txt")));

fn is_vowel(ch: u16) -> bool {
    matches!(ch, 0x61 | 0x65 | 0x69 | 0x6F | 0x75 | 0x79)
}

fn any_end(s: &[u16], len: usize, xs: &[&str]) -> bool {
    xs.iter().any(|x| ends_with(s, len, x))
}

/// `HungarianLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct HungarianLightStemmer;

impl HungarianLightStemmer {
    // Java: HungarianLightStemmer.removeCase
    fn remove_case(s: &[u16], len: usize) -> usize {
        if len > 6 && ends_with(s, len, "kent") {
            return len - 4;
        }
        if len > 5 {
            if any_end(
                s,
                len,
                &[
                    "nak", "nek", "val", "vel", "ert", "rol", "ban", "ben", "bol", "nal", "nel",
                    "hoz", "hez", "tol",
                ],
            ) {
                return len - 3;
            }
            if any_end(s, len, &["al", "el"]) && !is_vowel(s[len - 3]) && s[len - 3] == s[len - 4] {
                return len - 3;
            }
        }
        if len > 4 {
            if any_end(
                s,
                len,
                &[
                    "at", "et", "ot", "va", "ve", "ra", "re", "ba", "be", "ul", "ig",
                ],
            ) {
                return len - 2;
            }
            if any_end(s, len, &["on", "en"]) && !is_vowel(s[len - 3]) {
                return len - 2;
            }
            match s[len - 1] {
                0x74 | 0x6E => return len - 1,
                0x61 | 0x65 if s[len - 2] == s[len - 3] && !is_vowel(s[len - 2]) => return len - 2,
                _ => {}
            }
        }
        len
    }

    // Java: HungarianLightStemmer.removePossessive
    fn remove_possessive(s: &[u16], len: usize) -> usize {
        if len > 6 {
            if !is_vowel(s[len - 5]) && any_end(s, len, &["atok", "otok", "etek"]) {
                return len - 4;
            }
            if any_end(s, len, &["itek", "itok"]) {
                return len - 4;
            }
        }
        if len > 5 {
            if !is_vowel(s[len - 4]) && any_end(s, len, &["unk", "tok", "tek"]) {
                return len - 3;
            }
            if is_vowel(s[len - 4]) && ends_with(s, len, "juk") {
                return len - 3;
            }
            if ends_with(s, len, "ink") {
                return len - 3;
            }
        }
        if len > 4 {
            if !is_vowel(s[len - 3]) && any_end(s, len, &["am", "em", "om", "ad", "ed", "od", "uk"])
            {
                return len - 2;
            }
            if is_vowel(s[len - 3]) && any_end(s, len, &["nk", "ja", "je"]) {
                return len - 2;
            }
            if any_end(s, len, &["im", "id", "ik"]) {
                return len - 2;
            }
        }
        if len > 3 {
            match s[len - 1] {
                0x61 | 0x65 if !is_vowel(s[len - 2]) => return len - 1,
                0x6D | 0x64 if is_vowel(s[len - 2]) => return len - 1,
                0x69 => return len - 1,
                _ => {}
            }
        }
        len
    }

    // Java: HungarianLightStemmer.removePlural (with its fallthrough)
    fn remove_plural(s: &[u16], len: usize) -> usize {
        if len > 3 && s[len - 1] == u16::from(b'k') {
            if matches!(s[len - 2], 0x61 | 0x6F | 0x65) && len > 4 {
                return len - 2;
            }
            return len - 1;
        }
        len
    }

    // Java: HungarianLightStemmer.normalize
    fn normalize(s: &[u16], len: usize) -> usize {
        if len > 3 && matches!(s[len - 1], 0x61 | 0x65 | 0x69 | 0x6F) {
            return len - 1;
        }
        len
    }
}

impl CharStemmer for HungarianLightStemmer {
    // Java: HungarianLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE1 => u16::from(b'a'),
                0xEB | 0xE9 => u16::from(b'e'),
                0xED => u16::from(b'i'),
                0xF3 | 0x151 | 0xF5 | 0xF6 => u16::from(b'o'),
                0xFA | 0x171 | 0x169 | 0xFB | 0xFC => u16::from(b'u'),
                o => o,
            };
        }
        let len = Self::remove_case(s, len);
        let len = Self::remove_possessive(s, len);
        let len = Self::remove_plural(s, len);
        Self::normalize(s, len)
    }
}

/// `HungarianLightStemFilter`.
pub type HungarianLightStemFilter<I> = StemFilter<I, HungarianLightStemmer>;

language_analyzer! {
    /// `HungarianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `HungarianStemmer`.
    HungarianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Hungarian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
