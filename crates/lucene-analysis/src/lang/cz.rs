//! `org.apache.lucene.analysis.cz`: `CzechAnalyzer` and `CzechStemmer`
//! (Dolamic & Savoy's light stemmer).

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::ends_with;
use crate::CharArraySet;

use super::{comment_set, mark_exclusions, std_lower_stop, CharStemmer, StemFilter};

/// `CzechAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/cz_stopwords.txt")));

const fn c(ch: char) -> u16 {
    ch as u16
}

const CASE_3: [&str; 25] = [
    "ech", "ich", "ích", "ého", "ěmi", "emi", "ému", "ěte", "ete", "ěti", "eti", "ího", "iho",
    "ími", "ímu", "imu", "ách", "ata", "aty", "ých", "ama", "ami", "ové", "ovi", "ými",
];
const CASE_2: [&str; 12] = [
    "em", "es", "ém", "ím", "ům", "at", "ám", "os", "us", "ým", "mi", "ou",
];

/// `CzechStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CzechStemmer;

impl CzechStemmer {
    fn remove_case(s: &[u16], len: usize) -> usize {
        let any = |xs: &[&str]| xs.iter().any(|x| ends_with(s, len, x));
        if len > 7 && ends_with(s, len, "atech") {
            return len - 5;
        }
        if len > 6 && any(&["ětem", "etem", "atům"]) {
            return len - 4;
        }
        if len > 5 && any(&CASE_3) {
            return len - 3;
        }
        if len > 4 && any(&CASE_2) {
            return len - 2;
        }
        if len > 3
            && [
                c('a'),
                c('e'),
                c('i'),
                c('o'),
                c('u'),
                c('ů'),
                c('y'),
                c('á'),
                c('é'),
                c('í'),
                c('ý'),
                c('ě'),
            ]
            .contains(&s[len - 1])
        {
            return len - 1;
        }
        len
    }

    fn remove_possessives(s: &[u16], len: usize) -> usize {
        if len > 5 && ["ov", "in", "ův"].iter().any(|x| ends_with(s, len, x)) {
            return len - 2;
        }
        len
    }

    fn normalize(s: &mut [u16], len: usize) -> usize {
        if ends_with(s, len, "čt") {
            s[len - 2] = c('c');
            s[len - 1] = c('k');
            return len;
        }
        if ends_with(s, len, "št") {
            s[len - 2] = c('s');
            s[len - 1] = c('k');
            return len;
        }
        match s[len - 1] {
            0x63 | 0x10D => {
                s[len - 1] = c('k');
                return len;
            }
            0x7A | 0x17E => {
                s[len - 1] = c('h');
                return len;
            }
            _ => {}
        }
        if len > 1 && s[len - 2] == c('e') {
            s[len - 2] = s[len - 1];
            return len - 1;
        }
        if len > 2 && s[len - 2] == c('ů') {
            s[len - 2] = c('o');
        }
        len
    }
}

impl CharStemmer for CzechStemmer {
    // Java: CzechStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        let len = Self::remove_case(s, len);
        let len = Self::remove_possessives(s, len);
        if len > 0 {
            Self::normalize(s, len)
        } else {
            len
        }
    }
}

/// `CzechStemFilter`.
pub type CzechStemFilter<I> = StemFilter<I, CzechStemmer>;

language_analyzer! {
    /// `CzechAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, [`CzechStemFilter`].
    CzechAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        CzechStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
