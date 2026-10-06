//! `org.apache.lucene.analysis.hy`: `ArmenianAnalyzer` (Snowball `ArmenianStemmer`).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{comment_set, mark_exclusions, snowball, std_lower_stop};

/// `ArmenianAnalyzer.getDefaultStopSet()` (`hy_stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/hy_stopwords.txt")));

language_analyzer! {
    /// `ArmenianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, Snowball `ArmenianStemmer`.
    ArmenianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Armenian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
