//! `org.apache.lucene.analysis.da`: `DanishAnalyzer` (Snowball `DanishStemmer`).

use std::sync::{Arc, LazyLock};

use crate::CharArraySet;

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop};

/// `DanishAnalyzer.getDefaultStopSet()` (`danish_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/danish_stop.txt")));

language_analyzer! {
    /// `DanishAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`, `StopFilter`,
    /// exclusions, Snowball `DanishStemmer`.
    DanishAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Danish")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
