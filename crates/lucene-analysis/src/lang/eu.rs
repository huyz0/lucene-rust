//! `org.apache.lucene.analysis.eu`: `BasqueAnalyzer` (Snowball `BasqueStemmer`).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{comment_set, mark_exclusions, snowball, std_lower_stop};

/// `BasqueAnalyzer.getDefaultStopSet()` (`eu_stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/eu_stopwords.txt")));

language_analyzer! {
    /// `BasqueAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, Snowball `BasqueStemmer`.
    BasqueAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Basque")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
