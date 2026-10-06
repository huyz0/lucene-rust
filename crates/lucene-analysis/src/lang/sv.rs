//! `org.apache.lucene.analysis.sv`: `SwedishAnalyzer` (Snowball) and the
//! light and minimal stemmers.

use std::sync::{Arc, LazyLock};

use crate::util::stemmer_util::ends_with;
use crate::CharArraySet;

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop, CharStemmer, StemFilter};

/// `SwedishAnalyzer.getDefaultStopSet()` (`snowball/swedish_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/swedish_stop.txt")));

fn any_end(s: &[u16], len: usize, suffixes: &[&str]) -> bool {
    suffixes.iter().any(|x| ends_with(s, len, x))
}

/// `SwedishLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwedishLightStemmer;

impl CharStemmer for SwedishLightStemmer {
    // Java: SwedishLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len > 4 && s[len - 1] == u16::from(b's') {
            len -= 1;
        }
        if len > 7 && any_end(s, len, &["elser", "heten"]) {
            return len - 5;
        }
        if len > 6
            && any_end(
                s,
                len,
                &["arne", "erna", "ande", "else", "aste", "orna", "aren"],
            )
        {
            return len - 4;
        }
        if len > 5 && any_end(s, len, &["are", "ast", "het"]) {
            return len - 3;
        }
        if len > 4 && any_end(s, len, &["ar", "er", "or", "en", "at", "te", "et"]) {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 0x74 | 0x61 | 0x65 | 0x6E) {
            return len - 1;
        }
        len
    }
}

/// `SwedishMinimalStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct SwedishMinimalStemmer;

impl CharStemmer for SwedishMinimalStemmer {
    // Java: SwedishMinimalStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len > 4 && s[len - 1] == u16::from(b's') {
            len -= 1;
        }
        if len > 6 && any_end(s, len, &["arne", "erna", "arna", "orna", "aren"]) {
            return len - 4;
        }
        if len > 5 && any_end(s, len, &["are"]) {
            return len - 3;
        }
        if len > 4 && any_end(s, len, &["ar", "at", "er", "et", "or", "en"]) {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 0x61 | 0x65 | 0x6E) {
            return len - 1;
        }
        len
    }
}

/// `SwedishLightStemFilter`.
pub type SwedishLightStemFilter<I> = StemFilter<I, SwedishLightStemmer>;
/// `SwedishMinimalStemFilter`.
pub type SwedishMinimalStemFilter<I> = StemFilter<I, SwedishMinimalStemmer>;

language_analyzer! {
    /// `SwedishAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `SwedishStemmer`.
    SwedishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Swedish")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
