//! `org.apache.lucene.analysis.fi`: `FinnishAnalyzer` (Snowball) and the
//! light stemmer.

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{delete, ends_with};
use crate::CharArraySet;

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop, CharStemmer, StemFilter};

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `FinnishAnalyzer.getDefaultStopSet()` (`snowball/finnish_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/finnish_stop.txt")));

fn is_vowel(ch: u16) -> bool {
    matches!(ch, 0x61 | 0x65 | 0x69 | 0x6F | 0x75 | 0x79)
}

fn any_end(s: &[u16], len: usize, xs: &[&str]) -> bool {
    xs.iter().any(|x| ends_with(s, len, x))
}

/// `FinnishLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FinnishLightStemmer;

impl FinnishLightStemmer {
    fn step1(s: &[u16], len: usize) -> usize {
        if len > 8 {
            if ends_with(s, len, "kin") {
                return Self::step1(s, len - 3);
            }
            if ends_with(s, len, "ko") {
                return Self::step1(s, len - 2);
            }
        }
        if len > 11 {
            if ends_with(s, len, "dellinen") {
                return len - 8;
            }
            if ends_with(s, len, "dellisuus") {
                return len - 9;
            }
        }
        len
    }

    fn step2(s: &[u16], len: usize) -> usize {
        if len > 5 {
            if any_end(s, len, &["lla", "tse", "sti"]) {
                return len - 3;
            }
            if ends_with(s, len, "ni") {
                return len - 2;
            }
            if ends_with(s, len, "aa") {
                return len - 1;
            }
        }
        len
    }

    fn step3(s: &mut [u16], len: usize) -> usize {
        if len > 8 {
            if ends_with(s, len, "nnen") {
                s[len - 4] = c('s');
                return len - 3;
            }
            if ends_with(s, len, "ntena") {
                s[len - 5] = c('s');
                return len - 4;
            }
            if ends_with(s, len, "tten") {
                return len - 4;
            }
            if ends_with(s, len, "eiden") {
                return len - 5;
            }
        }
        if len > 6 {
            if any_end(s, len, &["neen", "niin", "seen", "teen", "inen"]) {
                return len - 4;
            }
            if s[len - 3] == c('h') && is_vowel(s[len - 2]) && s[len - 1] == c('n') {
                return len - 3;
            }
            if ends_with(s, len, "den") {
                s[len - 3] = c('s');
                return len - 2;
            }
            if ends_with(s, len, "ksen") {
                s[len - 4] = c('s');
                return len - 3;
            }
            if any_end(s, len, &["ssa", "sta", "lla", "lta", "tta", "ksi", "lle"]) {
                return len - 3;
            }
        }
        if len > 5 {
            if any_end(s, len, &["na", "ne"]) {
                return len - 2;
            }
            if ends_with(s, len, "nei") {
                return len - 3;
            }
        }
        if len > 4 {
            if any_end(s, len, &["ja", "ta"]) {
                return len - 2;
            }
            if s[len - 1] == c('a') {
                return len - 1;
            }
            if s[len - 1] == c('n') && is_vowel(s[len - 2]) {
                return len - 2;
            }
            if s[len - 1] == c('n') {
                return len - 1;
            }
        }
        len
    }

    fn norm1(s: &mut [u16], len: usize) -> usize {
        if len > 5 && ends_with(s, len, "hde") {
            s[len - 3] = c('k');
            s[len - 2] = c('s');
            s[len - 1] = c('i');
        }
        if len > 4 && any_end(s, len, &["ei", "at"]) {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 0x74 | 0x73 | 0x6A | 0x65 | 0x61 | 0x69) {
            return len - 1;
        }
        len
    }

    fn norm2(s: &mut [u16], mut len: usize) -> usize {
        if len > 8 && matches!(s[len - 1], 0x65 | 0x6F | 0x75) {
            len -= 1;
        }
        if len > 4 {
            if s[len - 1] == c('i') {
                len -= 1;
            }
            if len > 4 {
                let mut ch = s[0];
                let mut i = 1;
                while i < len {
                    if s[i] == ch && matches!(ch, 0x6B | 0x70 | 0x74) {
                        len = delete(s, i, len);
                    } else {
                        ch = s[i];
                        i += 1;
                    }
                }
            }
        }
        len
    }
}

impl CharStemmer for FinnishLightStemmer {
    // Java: FinnishLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len < 4 {
            return len;
        }
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE4 | 0xE5 => c('a'),
                0xF6 => c('o'),
                o => o,
            };
        }
        len = Self::step1(s, len);
        len = Self::step2(s, len);
        len = Self::step3(s, len);
        len = Self::norm1(s, len);
        Self::norm2(s, len)
    }
}

/// `FinnishLightStemFilter`.
pub type FinnishLightStemFilter<I> = StemFilter<I, FinnishLightStemmer>;

language_analyzer! {
    /// `FinnishAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `FinnishStemmer`.
    FinnishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Finnish")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
