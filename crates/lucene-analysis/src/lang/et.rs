//! `org.apache.lucene.analysis.et`: `EstonianAnalyzer` (Snowball `EstonianStemmer`).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{comment_set, mark_exclusions, snowball, std_lower_stop};

/// `EstonianAnalyzer.getDefaultStopSet()` (`et_stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/et_stopwords.txt")));

language_analyzer! {
    /// `EstonianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, Snowball `EstonianStemmer`.
    EstonianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Estonian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
