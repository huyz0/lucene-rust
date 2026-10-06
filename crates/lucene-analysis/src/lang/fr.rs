//! `org.apache.lucene.analysis.fr`: `FrenchAnalyzer` and the light and
//! minimal stemmers (Jacques Savoy's algorithms).

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::{delete, ends_with};
use crate::util::ElisionFilter;
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::{mark_exclusions, snowball_set, CharStemmer, StemFilter};

const fn c(ch: char) -> u16 {
    ch as u16
}

fn is_letter(u: u16) -> bool {
    crate::java_character::is_letter(u32::from(u))
}

/// `FrenchAnalyzer.getDefaultStopSet()` (`snowball/french_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/french_stop.txt")));

/// `FrenchAnalyzer.DEFAULT_ARTICLES`, the elided articles.
pub static DEFAULT_ARTICLES: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
    Arc::new(CharArraySet::from_words(
        [
            "l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu", "puisqu",
        ],
        true,
    ))
});

/// `FrenchLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrenchLightStemmer;

impl FrenchLightStemmer {
    // Java: FrenchLightStemmer.norm
    fn norm(s: &mut [u16], mut len: usize) -> usize {
        if len > 4 {
            for ch in s[..len].iter_mut() {
                *ch = match *ch {
                    0xE0 | 0xE1 | 0xE2 => c('a'),
                    0xF4 => c('o'),
                    0xE8 | 0xE9 | 0xEA => c('e'),
                    0xF9 | 0xFB => c('u'),
                    0xEE => c('i'),
                    0xE7 => c('c'),
                    o => o,
                };
            }
            let mut ch = s[0];
            let mut i = 1;
            while i < len {
                if s[i] == ch && is_letter(ch) {
                    len = delete(s, i, len);
                } else {
                    ch = s[i];
                    i += 1;
                }
            }
        }
        if len > 4 && ends_with(s, len, "ie") {
            len -= 2;
        }
        if len > 4 {
            if s[len - 1] == c('r') {
                len -= 1;
            }
            if s[len - 1] == c('e') {
                len -= 1;
            }
            if s[len - 1] == c('e') {
                len -= 1;
            }
            if s[len - 1] == s[len - 2] && is_letter(s[len - 1]) {
                len -= 1;
            }
        }
        len
    }
}

impl CharStemmer for FrenchLightStemmer {
    // Java: FrenchLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let s = &mut s[..];
        let norm = Self::norm;
        if len > 5 && s[len - 1] == c('x') {
            if s[len - 3] == c('a') && s[len - 2] == c('u') && s[len - 4] != c('e') {
                s[len - 2] = c('l');
            }
            len -= 1;
        }
        if len > 3 && s[len - 1] == c('x') {
            len -= 1;
        }
        if len > 3 && s[len - 1] == c('s') {
            len -= 1;
        }
        if len > 9 && ends_with(s, len, "issement") {
            len -= 6;
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 8 && ends_with(s, len, "issant") {
            len -= 4;
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 6 && ends_with(s, len, "ement") {
            len -= 4;
            if len > 3 && ends_with(s, len, "ive") {
                len -= 1;
                s[len - 1] = c('f');
            }
            return norm(s, len);
        }
        if len > 11 && ends_with(s, len, "ficatrice") {
            len -= 5;
            s[len - 2] = c('e');
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 10 && ends_with(s, len, "ficateur") {
            len -= 4;
            s[len - 2] = c('e');
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 9 && ends_with(s, len, "catrice") {
            len -= 3;
            s[len - 4] = c('q');
            s[len - 3] = c('u');
            s[len - 2] = c('e');
            return norm(s, len);
        }
        if len > 8 && ends_with(s, len, "cateur") {
            len -= 2;
            s[len - 4] = c('q');
            s[len - 3] = c('u');
            s[len - 2] = c('e');
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 8 && ends_with(s, len, "atrice") {
            len -= 4;
            s[len - 2] = c('e');
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 7 && ends_with(s, len, "ateur") {
            len -= 3;
            s[len - 2] = c('e');
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 6 && ends_with(s, len, "trice") {
            len -= 1;
            s[len - 3] = c('e');
            s[len - 2] = c('u');
            s[len - 1] = c('r');
        }
        if len > 5 && ends_with(s, len, "ième") {
            return norm(s, len - 4);
        }
        if len > 7 && ends_with(s, len, "teuse") {
            len -= 2;
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 6 && ends_with(s, len, "teur") {
            len -= 1;
            s[len - 1] = c('r');
            return norm(s, len);
        }
        if len > 5 && ends_with(s, len, "euse") {
            return norm(s, len - 2);
        }
        if len > 8 && ends_with(s, len, "ère") {
            len -= 1;
            s[len - 2] = c('e');
            return norm(s, len);
        }
        if len > 7 && ends_with(s, len, "ive") {
            len -= 1;
            s[len - 1] = c('f');
            return norm(s, len);
        }
        if len > 4 && (ends_with(s, len, "folle") || ends_with(s, len, "molle")) {
            len -= 2;
            s[len - 1] = c('u');
            return norm(s, len);
        }
        if len > 9 && ends_with(s, len, "nnelle") {
            return norm(s, len - 5);
        }
        if len > 9 && ends_with(s, len, "nnel") {
            return norm(s, len - 3);
        }
        if len > 4 && ends_with(s, len, "ète") {
            len -= 1;
            s[len - 2] = c('e');
        }
        if len > 8 && ends_with(s, len, "ique") {
            len -= 4;
        }
        if len > 8 && ends_with(s, len, "esse") {
            return norm(s, len - 3);
        }
        if len > 7 && ends_with(s, len, "inage") {
            return norm(s, len - 3);
        }
        if len > 9 && ends_with(s, len, "isation") {
            len -= 7;
            if len > 5 && ends_with(s, len, "ual") {
                s[len - 2] = c('e');
            }
            return norm(s, len);
        }
        if len > 9 && ends_with(s, len, "isateur") {
            return norm(s, len - 7);
        }
        if len > 8 && (ends_with(s, len, "ation") || ends_with(s, len, "ition")) {
            return norm(s, len - 5);
        }
        norm(s, len)
    }
}

/// `FrenchMinimalStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FrenchMinimalStemmer;

impl CharStemmer for FrenchMinimalStemmer {
    // Java: FrenchMinimalStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len < 6 {
            return len;
        }
        if s[len - 1] == c('x') {
            if s[len - 3] == c('a') && s[len - 2] == c('u') {
                s[len - 2] = c('l');
            }
            return len - 1;
        }
        for ch in ['s', 'r', 'e', 'é'] {
            if s[len - 1] == c(ch) {
                len -= 1;
            }
        }
        if s[len - 1] == s[len - 2] && is_letter(s[len - 1]) {
            len -= 1;
        }
        len
    }
}

/// `FrenchLightStemFilter`.
pub type FrenchLightStemFilter<I> = StemFilter<I, FrenchLightStemmer>;
/// `FrenchMinimalStemFilter`.
pub type FrenchMinimalStemFilter<I> = StemFilter<I, FrenchMinimalStemmer>;

language_analyzer! {
    /// `FrenchAnalyzer`: `StandardTokenizer`, `ElisionFilter`,
    /// `LowerCaseFilter`, `StopFilter`, exclusions, [`FrenchLightStemFilter`].
    FrenchAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = StopFilter::new(
            LowerCaseFilter::new(ElisionFilter::new(
                StandardTokenizer::new(),
                Arc::clone(&DEFAULT_ARTICLES),
            )),
            Arc::clone(&s.stopwords),
        );
        FrenchLightStemFilter::new(mark_exclusions(r, &s.exclusion))
    }
    normalize(s, input) {
        LowerCaseFilter::new(ElisionFilter::new(input, Arc::clone(&DEFAULT_ARTICLES)))
    }
}
