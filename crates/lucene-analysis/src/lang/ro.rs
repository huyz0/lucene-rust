//! `org.apache.lucene.analysis.ro`: `RomanianAnalyzer` and
//! `RomanianNormalizationFilter` (cedilla `ş`/`ţ` to comma-below `ș`/`ț`).

use std::sync::{Arc, LazyLock};

use crate::{CharArraySet, LowerCaseFilter};

use super::{comment_set, mark_exclusions, snowball, std_lower_stop, CharStemmer, NormalizeFilter};

/// `RomanianAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/ro_stopwords.txt")));

/// `RomanianNormalizer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct RomanianNormalizer;

impl CharStemmer for RomanianNormalizer {
    // Java: RomanianNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        for ch in s[..len].iter_mut() {
            *ch = match *ch {
                0x15E => 0x218,
                0x15F => 0x219,
                0x162 => 0x21A,
                0x163 => 0x21B,
                o => o,
            };
        }
        len
    }
}

/// `RomanianNormalizationFilter`.
pub type RomanianNormalizationFilter<I> = NormalizeFilter<I, RomanianNormalizer>;

language_analyzer! {
    /// `RomanianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, [`RomanianNormalizationFilter`], exclusions, Snowball
    /// `RomanianStemmer`.
    RomanianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = RomanianNormalizationFilter::new(std_lower_stop(&s.stopwords));
        snowball(mark_exclusions(r, &s.exclusion), "Romanian")
    }
    normalize(s, input) {
        RomanianNormalizationFilter::new(LowerCaseFilter::new(input))
    }
}
