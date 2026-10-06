//! `org.apache.lucene.analysis.gl`: `GalicianAnalyzer` and the RSLP-based
//! Galician stemmers.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::pt::{parse_rslp, Step};
use super::{mark_exclusions, plain_set, std_lower_stop, CharStemmer, StemFilter};

/// `GalicianAnalyzer.getDefaultStopSet()`: `stopwords.txt` read with
/// `WordlistLoader.getWordSet` (no comment syntax, so its `#` header line is
/// a "word" too, as in Lucene).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| plain_set(include_str!("stopwords/gl_stopwords.txt")));

static STEPS: LazyLock<HashMap<String, Step>> =
    LazyLock::new(|| parse_rslp(include_str!("stopwords/galician.rslp")));

fn step(name: &str) -> &'static Step {
    &STEPS[name]
}

/// `GalicianStemmer` (RSLP-GL).
#[derive(Debug, Default, Clone, Copy)]
pub struct GalicianStemmer;

impl CharStemmer for GalicianStemmer {
    // Java: GalicianStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        for name in ["Plural", "Unification", "Adverb"] {
            len = step(name).apply(s, len);
        }
        loop {
            let old = len;
            len = step("Augmentative").apply(s, len);
            if len == old {
                break;
            }
        }
        let old = len;
        len = step("Noun").apply(s, len);
        if len == old {
            len = step("Verb").apply(s, len);
        }
        len = step("Vowel").apply(s, len);
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0xE1 => u16::from(b'a'),
                0xE9 | 0xEA => u16::from(b'e'),
                0xED => u16::from(b'i'),
                0xF3 => u16::from(b'o'),
                0xFA => u16::from(b'u'),
                o => o,
            };
        }
        len
    }
}

/// `GalicianMinimalStemmer`: RSLP-GL's plural step.
#[derive(Debug, Default, Clone, Copy)]
pub struct GalicianMinimalStemmer;

impl CharStemmer for GalicianMinimalStemmer {
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        step("Plural").apply(s, len)
    }
}

/// `GalicianStemFilter`.
pub type GalicianStemFilter<I> = StemFilter<I, GalicianStemmer>;
/// `GalicianMinimalStemFilter`.
pub type GalicianMinimalStemFilter<I> = StemFilter<I, GalicianMinimalStemmer>;

language_analyzer! {
    /// `GalicianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, [`GalicianStemFilter`].
    GalicianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        GalicianStemFilter::new(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion))
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
