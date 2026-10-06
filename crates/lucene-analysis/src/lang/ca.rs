//! `org.apache.lucene.analysis.ca`: `CatalanAnalyzer` (elision, Snowball
//! `CatalanStemmer`).

use std::sync::{Arc, LazyLock};

use crate::util::ElisionFilter;
use crate::{CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

use super::{comment_set, mark_exclusions, snowball};

/// `CatalanAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/ca_stopwords.txt")));

/// `CatalanAnalyzer.DEFAULT_ARTICLES`.
pub static DEFAULT_ARTICLES: LazyLock<Arc<CharArraySet>> = LazyLock::new(|| {
    Arc::new(CharArraySet::from_words(
        ["d", "l", "m", "n", "s", "t"],
        true,
    ))
});

language_analyzer! {
    /// `CatalanAnalyzer`: `StandardTokenizer`, `ElisionFilter`,
    /// `LowerCaseFilter`, `StopFilter`, exclusions, Snowball `CatalanStemmer`.
    CatalanAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        let r = StopFilter::new(
            LowerCaseFilter::new(ElisionFilter::new(
                StandardTokenizer::new(),
                Arc::clone(&DEFAULT_ARTICLES),
            )),
            Arc::clone(&s.stopwords),
        );
        snowball(mark_exclusions(r, &s.exclusion), "Catalan")
    }
    normalize(s, input) {
        LowerCaseFilter::new(ElisionFilter::new(input, Arc::clone(&DEFAULT_ARTICLES)))
    }
}
