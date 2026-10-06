//! `org.apache.lucene.analysis.it`: `ItalianAnalyzer` and the light stemmer.

use std::sync::{Arc, LazyLock};

use crate::util::ElisionFilter;
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::es::fold_vowels;
use super::{mark_exclusions, snowball_set, CharStemmer, StemFilter};

/// `ItalianAnalyzer.getDefaultStopSet()` (`snowball/italian_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/italian_stop.txt")));

/// `ItalianAnalyzer.DEFAULT_ARTICLES`.
pub static DEFAULT_ARTICLES: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
    Arc::new(CharArraySet::from_words(
        [
            "c", "l", "all", "dall", "dell", "nell", "sull", "coll", "pell", "gl", "agl", "dagl",
            "degl", "negl", "sugl", "un", "m", "t", "s", "v", "d",
        ],
        true,
    ))
});

/// `ItalianLightStemmer`.
#[derive(Debug, Default, Clone, Copy)]
pub struct ItalianLightStemmer;

impl CharStemmer for ItalianLightStemmer {
    // Java: ItalianLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
        if len < 6 {
            return len;
        }
        fold_vowels(&mut s[..len]);
        let p = s[len - 2];
        let (i, h) = (u16::from(b'i'), u16::from(b'h'));
        match s[len - 1] {
            0x65 | 0x69 if p == i || p == h => len - 2,
            0x61 | 0x6F if p == i => len - 2,
            0x65 | 0x69 | 0x61 | 0x6F => len - 1,
            _ => len,
        }
    }
}

/// `ItalianLightStemFilter`.
pub type ItalianLightStemFilter<I> = StemFilter<I, ItalianLightStemmer>;

language_analyzer! {
    /// `ItalianAnalyzer`: `StandardTokenizer`, `ElisionFilter`,
    /// `LowerCaseFilter`, `StopFilter`, exclusions, [`ItalianLightStemFilter`].
    ItalianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = StopFilter::new(
            LowerCaseFilter::new(ElisionFilter::new(
                StandardTokenizer::new(),
                Arc::clone(&DEFAULT_ARTICLES),
            )),
            Arc::clone(&s.stopwords),
        );
        ItalianLightStemFilter::new(mark_exclusions(r, &s.exclusion))
    }
    normalize(s, input) {
        LowerCaseFilter::new(ElisionFilter::new(input, Arc::clone(&DEFAULT_ARTICLES)))
    }
}
