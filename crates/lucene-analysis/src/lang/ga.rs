//! `org.apache.lucene.analysis.ga`: `IrishAnalyzer` and
//! `IrishLowerCaseFilter` (`nA` -> `n-a`, then lowercase).

use std::sync::{Arc, LazyLock};

use crate::java_character::to_lower_case;
use crate::util::ElisionFilter;
use crate::{CharArraySet, StandardTokenizer, StopFilter};

use super::{mark_exclusions, snowball, snowball_set, CharStemmer, NormalizeFilter};

/// `IrishAnalyzer.getDefaultStopSet()` (`snowball/irish_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/irish_stop.txt")));

/// `IrishAnalyzer.DEFAULT_ARTICLES`.
pub static DEFAULT_ARTICLES: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| Arc::new(CharArraySet::from_words(["d", "m", "b"], true)));

/// `IrishAnalyzer.HYPHENATIONS`: `t`/`n`/`h` split off by a hyphen.
pub static HYPHENATIONS: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| Arc::new(CharArraySet::from_words(["h", "n", "t"], true)));

/// `IrishLowerCaseFilter`'s per-term transform.
#[derive(Debug, Default, Clone, Copy)]
pub struct IrishLowerCase;

fn is_upper_vowel(v: u16) -> bool {
    matches!(
        v,
        0x41 | 0x45 | 0x49 | 0x4F | 0x55 | 0xC1 | 0xC9 | 0xCD | 0xD3 | 0xDA
    )
}

impl CharStemmer for IrishLowerCase {
    // Java: IrishLowerCaseFilter.incrementToken
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let mut idx = 0;
        if len > 1 && (s[0] == u16::from(b'n') || s[0] == u16::from(b't')) && is_upper_vowel(s[1]) {
            s.insert(1, u16::from(b'-'));
            idx = 2;
            len += 1;
        }
        // `Character.toLowerCase(char)`: per UTF-16 unit.
        for ch in s[idx..len].iter_mut() {
            *ch = to_lower_case(u32::from(*ch)) as u16;
        }
        len
    }
}

/// `IrishLowerCaseFilter`.
pub type IrishLowerCaseFilter<I> = NormalizeFilter<I, IrishLowerCase>;

language_analyzer! {
    /// `IrishAnalyzer`: `StandardTokenizer`, `StopFilter(HYPHENATIONS)`,
    /// `ElisionFilter`, [`IrishLowerCaseFilter`], `StopFilter`, exclusions,
    /// Snowball `IrishStemmer`.
    IrishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = StopFilter::new(StandardTokenizer::new(), Arc::clone(&HYPHENATIONS));
        let r = IrishLowerCaseFilter::new(ElisionFilter::new(r, Arc::clone(&DEFAULT_ARTICLES)));
        let r = StopFilter::new(r, Arc::clone(&s.stopwords));
        snowball(mark_exclusions(r, &s.exclusion), "Irish")
    }
    normalize(s, input) {
        IrishLowerCaseFilter::new(ElisionFilter::new(input, Arc::clone(&DEFAULT_ARTICLES)))
    }
}
