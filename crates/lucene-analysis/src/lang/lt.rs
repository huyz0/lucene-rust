//! `org.apache.lucene.analysis.lt`: `LithuanianAnalyzer` (Snowball `LithuanianStemmer`).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{comment_set, mark_exclusions, snowball, std_lower_stop};

/// `LithuanianAnalyzer.getDefaultStopSet()` (`lt_stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/lt_stopwords.txt")));

language_analyzer! {
    /// `LithuanianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, Snowball `LithuanianStemmer`.
    LithuanianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Lithuanian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
