//! `org.apache.lucene.analysis.ru`: `RussianAnalyzer` (Snowball) and the
//! light stemmer.

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{ends_with_units, utf16};
use crate::CharArraySet;

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop, CharStemmer, StemFilter};

/// `RussianAnalyzer.getDefaultStopSet()` (`snowball/russian_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/russian_stop.txt")));

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `RussianLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct RussianLightStemmer;

const ENDINGS_4: [&[u16]; 2] = [utf16!("иями"), utf16!("оями")];
const ENDINGS_3: [&[u16]; 15] = [
    utf16!("иям"),
    utf16!("иях"),
    utf16!("оях"),
    utf16!("ями"),
    utf16!("оям"),
    utf16!("оьв"),
    utf16!("ами"),
    utf16!("его"),
    utf16!("ему"),
    utf16!("ери"),
    utf16!("ими"),
    utf16!("ого"),
    utf16!("ому"),
    utf16!("ыми"),
    utf16!("оев"),
];
const ENDINGS_2: [&[u16]; 30] = [
    utf16!("ая"),
    utf16!("яя"),
    utf16!("ях"),
    utf16!("юю"),
    utf16!("ах"),
    utf16!("ею"),
    utf16!("их"),
    utf16!("ия"),
    utf16!("ию"),
    utf16!("ьв"),
    utf16!("ою"),
    utf16!("ую"),
    utf16!("ям"),
    utf16!("ых"),
    utf16!("ея"),
    utf16!("ам"),
    utf16!("ем"),
    utf16!("ей"),
    utf16!("ём"),
    utf16!("ев"),
    utf16!("ий"),
    utf16!("им"),
    utf16!("ое"),
    utf16!("ой"),
    utf16!("ом"),
    utf16!("ов"),
    utf16!("ые"),
    utf16!("ый"),
    utf16!("ым"),
    utf16!("ми"),
];

impl RussianLightStemmer {
    // Java: RussianLightStemmer.removeCase
    fn remove_case(s: &[u16], len: usize) -> usize {
        let any = |xs: &[&[u16]]| xs.iter().any(|x| ends_with_units(s, len, x));
        if len > 6 && any(&ENDINGS_4) {
            return len - 4;
        }
        if len > 5 && any(&ENDINGS_3) {
            return len - 3;
        }
        if len > 4 && any(&ENDINGS_2) {
            return len - 2;
        }
        if len > 3
            && [
                c('а'),
                c('е'),
                c('и'),
                c('о'),
                c('у'),
                c('й'),
                c('ы'),
                c('я'),
                c('ь'),
            ]
            .contains(&s[len - 1])
        {
            return len - 1;
        }
        len
    }

    // Java: RussianLightStemmer.normalize
    fn normalize(s: &[u16], len: usize) -> usize {
        if len > 3 {
            let last = s[len - 1];
            if last == c('ь') || last == c('и') || (last == c('н') && s[len - 2] == c('н')) {
                return len - 1;
            }
        }
        len
    }
}

impl CharStemmer for RussianLightStemmer {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let len = Self::remove_case(s, len);
        Self::normalize(s, len)
    }
}

/// `RussianLightStemFilter`.
pub type RussianLightStemFilter<I> = StemFilter<I, RussianLightStemmer>;

language_analyzer! {
    /// `RussianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `RussianStemmer`.
    RussianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Russian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
