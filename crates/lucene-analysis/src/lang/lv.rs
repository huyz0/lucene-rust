//! `org.apache.lucene.analysis.lv`: `LatvianAnalyzer` and `LatvianStemmer`
//! (Kreslins' light stemmer).

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::ends_with;
use crate::CharArraySet;

use super::{mark_exclusions, plain_set, std_lower_stop, CharStemmer, StemFilter};

/// `LatvianAnalyzer.getDefaultStopSet()`: `stopwords.txt` read with
/// `WordlistLoader.getWordSet` (no comment syntax).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| plain_set(include_str!("stopwords/lv_stopwords.txt")));

const fn c(ch: char) -> u16 {
    ch as u16
}

/// `LatvianStemmer.affixes`: suffix, vowel count, palatalizes.
const AFFIXES: [(&str, usize, bool); 38] = [
    ("ajiem", 3, false),
    ("ajai", 3, false),
    ("ajam", 2, false),
    ("ajām", 2, false),
    ("ajos", 2, false),
    ("ajās", 2, false),
    ("iem", 2, true),
    ("ajā", 2, false),
    ("ais", 2, false),
    ("ai", 2, false),
    ("ei", 2, false),
    ("ām", 1, false),
    ("am", 1, false),
    ("ēm", 1, false),
    ("īm", 1, false),
    ("im", 1, false),
    ("um", 1, false),
    ("us", 1, true),
    ("as", 1, false),
    ("ās", 1, false),
    ("es", 1, false),
    ("os", 1, true),
    ("ij", 1, false),
    ("īs", 1, false),
    ("ēs", 1, false),
    ("is", 1, false),
    ("ie", 1, false),
    ("u", 1, true),
    ("a", 1, true),
    ("i", 1, true),
    ("e", 1, false),
    ("ā", 1, false),
    ("ē", 1, false),
    ("ī", 1, false),
    ("ū", 1, false),
    ("o", 1, false),
    ("s", 0, false),
    ("š", 0, false),
];

/// `LatvianStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct LatvianStemmer;

impl LatvianStemmer {
    // Java: LatvianStemmer.unpalatalize (reads s[len], the removed suffix's
    // first char, still in the buffer)
    fn unpalatalize(s: &mut [u16], mut len: usize) -> usize {
        if s[len] == c('u') {
            if ends_with(s, len, "kš") {
                len += 1;
                s[len - 2] = c('s');
                s[len - 1] = c('t');
                return len;
            }
            if ends_with(s, len, "ņņ") {
                s[len - 2] = c('n');
                s[len - 1] = c('n');
                return len;
            }
        }
        if ["pj", "bj", "mj", "vj"]
            .iter()
            .any(|x| ends_with(s, len, x))
        {
            return len - 1;
        }
        for (x, a, b) in [
            ("šņ", 's', 'n'),
            ("žņ", 'z', 'n'),
            ("šļ", 's', 'l'),
            ("žļ", 'z', 'l'),
            ("ļņ", 'l', 'n'),
            ("ļļ", 'l', 'l'),
        ] {
            if ends_with(s, len, x) {
                s[len - 2] = c(a);
                s[len - 1] = c(b);
                return len;
            }
        }
        let last = &mut s[len - 1];
        *last = match *last {
            0x10D => c('c'),
            0x13C => c('l'),
            0x146 => c('n'),
            o => o,
        };
        len
    }
}

impl CharStemmer for LatvianStemmer {
    // Java: LatvianStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let num_vowels = s[..len]
            .iter()
            .filter(|&&u| {
                [
                    c('a'),
                    c('e'),
                    c('i'),
                    c('o'),
                    c('u'),
                    c('ā'),
                    c('ī'),
                    c('ē'),
                    c('ū'),
                ]
                .contains(&u)
            })
            .count();
        for (affix, vc, palatalizes) in AFFIXES {
            let n = affix.encode_utf16().count();
            if num_vowels > vc && len >= n + 3 && ends_with(s, len, affix) {
                let len = len - n;
                return if palatalizes {
                    Self::unpalatalize(s, len)
                } else {
                    len
                };
            }
        }
        len
    }
}

/// `LatvianStemFilter`.
pub type LatvianStemFilter<I> = StemFilter<I, LatvianStemmer>;

language_analyzer! {
    /// `LatvianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`LatvianStemFilter`].
    LatvianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        LatvianStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
