//! `org.apache.lucene.analysis.ru`: `RussianAnalyzer` (Snowball) and the
//! light stemmer.

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::ends_with;
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

const ENDINGS_4: [&str; 2] = ["иями", "оями"];
const ENDINGS_3: [&str; 15] = [
    "иям", "иях", "оях", "ями", "оям", "оьв", "ами", "его", "ему", "ери", "ими", "ого", "ому",
    "ыми", "оев",
];
const ENDINGS_2: [&str; 30] = [
    "ая", "яя", "ях", "юю", "ах", "ею", "их", "ия", "ию", "ьв", "ою", "ую", "ям", "ых", "ея", "ам",
    "ем", "ей", "ём", "ев", "ий", "им", "ое", "ой", "ом", "ов", "ые", "ый", "ым", "ми",
];

impl RussianLightStemmer {
    // Java: RussianLightStemmer.removeCase
    fn remove_case(s: &[u16], len: usize) -> usize {
        let any = |xs: &[&str]| xs.iter().any(|x| ends_with(s, len, x));
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
